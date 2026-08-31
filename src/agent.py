"""
agent.py — the tool-calling investigation loop.

One run: give the model a datalog path, the gate report, and a set of
tool schemas; let it call query functions until it stops or reaches the
turn cap; return its report plus per-run metrics.

The loop is a plain reason-act cycle. Nothing here decides anything
about the data -- it only routes calls between the model and
analyze.py, and records what happened.
"""

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic

import tools

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

MODEL = "claude-haiku-4-5-20251001"
MAX_TURNS = 16
MAX_TOKENS = 4000
RETRIES = 3            # transient API failures, per turn
WRAP_UP = 3            # warn this many turns before the cap

# Appended to the tool results when the turn budget is nearly spent. An
# investigation that never terminates yields no answer to score, and on
# the control dataset -- where there is nothing to find -- the agent will
# otherwise keep slicing indefinitely. Applied identically in every
# configuration, so it cannot favour one over another.
WRAP_NOTE = ("[harness] {left} of {total} iterations remain. Write your "
             "final report now, beginning with the ROOT_CAUSE_DIMENSIONS "
             "line. Report only what your retrieved numbers support; NONE "
             "is a valid and expected answer when no systematic pattern "
             "is present.")

DIMENSIONS = ["LOT_ID", "WAFER_ID", "SITE_NUM", "TEMP_C", "VDD", "SPATIAL"]

SYSTEM = """You are a product engineer investigating a possible yield \
excursion in post-silicon ATE data.

The data has one row per test execution: one test, on one die, at one set \
of conditions. A die is scrapped if any of its measurements falls outside \
its limits.

Investigate using the tools provided. Form a hypothesis, then call a tool \
to test it before concluding. Do not speculate without data.

Every claim in your conclusion must cite specific numbers you retrieved.

Correlation in test data does not establish physical root cause. Say so, \
and recommend a next investigation step rather than asserting a cause.

If the data does not support a systematic failure pattern, say so plainly. \
Do not manufacture a finding.

Begin your final message with exactly this line, before anything else:

ROOT_CAUSE_DIMENSIONS: <comma-separated list, or NONE>

Valid dimensions: LOT_ID, WAFER_ID, SITE_NUM, TEMP_C, VDD, SPATIAL.
Use SPATIAL for any effect tied to die position on the wafer. List only \
dimensions your retrieved numbers support. Use NONE if there is no \
systematic pattern.

Below that line, give the report under these four headings, in order:

FINDINGS
  Numbered. One line each. Every line must contain the numbers you
  retrieved and the tool call they came from.

RULED OUT
  Dimensions you checked that did not explain the failures, with the
  numbers that eliminated them. An empty section here means you did not
  eliminate anything.

ASSESSMENT
  What the evidence is consistent with, and how confident you are.
  State explicitly whether you were able to quantify the combined
  effect or only observe the dimensions separately.

RECOMMENDED NEXT STEP
  What an engineer should do next. Note whether the pattern suggests an
  ongoing condition or one that has already resolved."""


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _parse_answer(text):
    """
    Pull the declared dimensions from the last ROOT_CAUSE_DIMENSIONS marker.

    Reads to the next blank line rather than the end of the line, because
    the model sometimes wraps a long list across two lines. Only tokens in
    DIMENSIONS are accepted, so any prose caught in the window is ignored.

    Returns a list of dimensions, [] for an explicit NONE, or None if the
    marker is absent. The last case is a different outcome from the second
    and is scored differently: the agent never answered in the required
    format, rather than answering that nothing was found.
    """
    marks = list(re.finditer(r"ROOT_CAUSE_DIMENSIONS:\s*", text or "",
                             re.IGNORECASE))
    if not marks:
        return None

    tail = text[marks[-1].end():]
    tail = re.split(r"\n\s*\n", tail)[0][:300]      # stop at a blank line

    if re.match(r"\s*NONE\b", tail, re.IGNORECASE):
        return []

    found = []
    for tok in re.findall(r"[A-Za-z_]+", tail):
        t = tok.upper()
        if t in DIMENSIONS and t not in found:
            found.append(t)
    return found


def _serialise(messages):
    """
    Convert the conversation into plain JSON-safe structures.

    The SDK returns content blocks as objects, which json.dump cannot
    handle. run_matrix.py writes one transcript per run, so this has to
    happen before the record leaves this module.
    """
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        blocks = []
        for b in content:
            if isinstance(b, dict):
                blocks.append(b)
            elif b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "name": b.name,
                               "input": b.input})
            else:
                blocks.append({"type": getattr(b, "type", "unknown")})
        out.append({"role": msg["role"], "content": blocks})
    return out


def _call(client, schemas, messages):
    """One API call, retrying transient failures with a backoff."""
    last = None
    for attempt in range(RETRIES):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                tools=schemas,
                messages=messages,
            )
        except (anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            last = e
            time.sleep(2 ** attempt * 2)
    raise last


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------

def run(csv, config, gate_report=None):
    """
    Run one investigation.

    Returns the declared dimensions, the metrics used in the results
    table, the final report, and the full transcript.
    """
    client = Anthropic()
    schemas = tools.build_tools(config)

    opening = f"A lot group has been flagged for review.\n\nDatalog: {csv}\n"
    if gate_report:
        opening += f"\nAutomated triage gate report:\n\n{gate_report}\n"
    opening += "\nInvestigate and report your conclusion."

    messages = [{"role": "user", "content": opening}]
    calls = []
    turns = 0
    stop = "max_turns"
    tok_in = tok_out = 0

    for turns in range(1, MAX_TURNS + 1):
        resp = _call(client, schemas, messages)
        tok_in += resp.usage.input_tokens
        tok_out += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            stop = resp.stop_reason
            break

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            out = tools.dispatch(block.name, block.input, csv, config)
            args = block.input or {}
            calls.append({
                "turn": turns,
                "tool": block.name,
                "args": args,
                "error": "error" in out,
            })
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(out),
            })

        left = MAX_TURNS - turns
        if 0 < left <= WRAP_UP:
            results.append({"type": "text",
                            "text": WRAP_NOTE.format(left=left,
                                                     total=MAX_TURNS)})
        messages.append({"role": "user", "content": results})

    # last assistant message carrying text
    final = ""
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        text = "\n".join(b.text for b in msg["content"]
                         if getattr(b, "type", None) == "text")
        if text.strip():
            final = text
            break

    filtered = [c for c in calls if c["args"].get("filters")]

    return {
        "config": config,
        "csv": csv,
        "dimensions": _parse_answer(final),
        "turns": turns,
        "tool_calls": len(calls),
        "tool_errors": sum(c["error"] for c in calls),
        "filtered_calls": len(filtered),
        "used_filters": bool(filtered),
        "distinct_tools": len({c["tool"] for c in calls}),
        "cited_numbers": bool(re.search(r"\d", final)),
        "stop_reason": stop,
        "warned": MAX_TURNS - turns <= WRAP_UP,
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "report": final,
        "calls": calls,
        "transcript": _serialise(messages),
    }


# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import report_pdf

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/defect_4d.csv"
    config = sys.argv[2] if len(sys.argv) > 2 else "full"
    dataset = Path(csv).stem

    gate = ROOT / "runs" / f"gate_{dataset}.txt"
    r = run(csv, config, gate_report=gate.read_text() if gate.exists() else None)

    # ---- run record, then the reviewable PDF ----
    RESULTS.mkdir(exist_ok=True)
    stem = f"{config}_{dataset}"
    rec_path = RESULTS / f"{stem}.json"
    rec_path.write_text(json.dumps({
        "config": config,
        "dataset": dataset,
        "metrics": {
            "declared": "|".join(r["dimensions"])
                        if r["dimensions"] is not None else "NO_ANSWER_LINE",
            "turns": r["turns"],
            "tool_calls": r["tool_calls"],
            "filtered_calls": r["filtered_calls"],
            "distinct_tools": r["distinct_tools"],
            "tool_errors": r["tool_errors"],
            "stop_reason": r["stop_reason"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
        },
        "calls": r["calls"],
        "report": r["report"],
        "transcript": r["transcript"],
    }, indent=2))

    pdf_path = report_pdf.build(rec_path, RESULTS / f"{stem}.pdf")

    # ---- short console summary; the report itself is in the PDF ----
    print(f"config        {config}")
    print(f"dataset       {dataset}")
    print(f"turns         {r['turns']}  ({r['stop_reason']})")
    print(f"tool calls    {r['tool_calls']}  "
          f"({r['filtered_calls']} filtered, {r['tool_errors']} errors)")
    print(f"distinct      {r['distinct_tools']} tools")
    print(f"tokens        {r['input_tokens']:,} in / {r['output_tokens']:,} out")
    print(f"dimensions    {r['dimensions']}")
    print()
    print(f"report        {pdf_path}")
    print(f"run record    {rec_path}")