"""
run_matrix.py — run the experiment and score every run.

4 configurations x 2 datasets x 5 repeats = 40 runs.

Writes results/results.csv (one row per run) and runs/<name>.json (one
transcript per run). PDFs are not generated here; render the ones you
want with:

    python src/report_pdf.py runs/full_defect_4d_0.json

Usage:
    python src/run_matrix.py                # all 40
    python src/run_matrix.py --reps 1       # smoke test, 8 runs
    python src/run_matrix.py --config full  # one configuration
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import agent

ROOT = Path(__file__).parent.parent
CONFIGS = ["minimal", "minimal_f", "reduced", "full"]

# Ground truth. WAFER_ID and SITE_NUM are in the answer space but in no
# ground truth, so declaring either counts as spurious.
TRUTH = {
    "clean":     set(),
    "defect_4d": {"LOT_ID", "SPATIAL", "TEMP_C", "VDD"},
}

# Values behind each dimension, as substrings to look for in what the
# agent declared. Scored separately from the dimensions: an agent can
# name the right axis and still get the value wrong.
TRUTH_VALUES = {
    "clean": {},
    "defect_4d": {
        "LOT_ID":  ["L47", "L48"],
        "TEMP_C":  ["85"],
        "VDD":     ["0.75"],
        "SPATIAL": ["NW"],
    },
}

FIELDS = ["config", "dataset", "rep", "declared", "values",
          "structure", "evidence", "structure_correct",
          "n_correct", "n_spurious", "exact", "missed",
          "n_values_correct", "values_exact",
          "turns", "tool_calls", "filtered_calls", "stop_reason",
          "warned", "cited_numbers", "input_tokens", "output_tokens",
          "seconds", "error"]


def score(declared, truth):
    """Compare the declared set against ground truth.

    declared is None when no answer line was emitted, which is a
    different outcome from declaring nothing and is left blank.
    """
    if declared is None:
        return {"n_correct": "", "n_spurious": "", "exact": "", "missed": ""}
    d = set(declared)
    return {"n_correct": len(d & truth),
            "n_spurious": len(d - truth),
            "exact": int(d == truth),
            "missed": "|".join(sorted(truth - d))}


def score_values(declared, truth):
    """
    Count how many dimensions were given the correct value.

    A dimension counts as correct if every expected value appears
    somewhere in what the agent wrote for it, so "edge NW" matches an
    expected "NW". Only dimensions present in the truth are checked.
    """
    if declared is None or not truth:
        return {"n_values_correct": "", "values_exact": ""}
    n = 0
    for dim, want in truth.items():
        got = " ".join(declared.get(dim, [])).upper()
        if got and all(w.upper() in got for w in want):
            n += 1
    return {"n_values_correct": n, "values_exact": int(n == len(truth))}


def one_run(config, dataset, rep):
    csv_path = ROOT / "data" / f"{dataset}.csv"
    gate = ROOT / "runs" / f"gate_{dataset}.txt"
    row = {"config": config, "dataset": dataset, "rep": rep, "error": ""}
    t0 = time.time()

    try:
        r = agent.run(str(csv_path), config,
                      gate_report=gate.read_text() if gate.exists() else None)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = round(time.time() - t0, 1)
        traceback.print_exc()
        return row, None

    row.update({
        "declared": "|".join(r["dimensions"]) if r["dimensions"] is not None
                    else "NO_ANSWER_LINE",
        "values": "; ".join(f"{k}={','.join(v)}" for k, v in
                            (r["values"] or {}).items()) or "",
        "structure": r["structure"] or "",
        "evidence": r["evidence"] or "",
        # the injected condition is confined to an intersection; on the
        # control the correct structural answer is none
        "structure_correct": int(r["structure"] ==
                                 ("interaction" if dataset == "defect_4d"
                                  else "none")) if r["structure"] else "",
        "turns": r["turns"],
        "tool_calls": r["tool_calls"],
        "filtered_calls": r["filtered_calls"],
        "stop_reason": r["stop_reason"],
        "warned": int(r["warned"]),
        "cited_numbers": int(r["cited_numbers"]),
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
        "seconds": round(time.time() - t0, 1),
    })
    row.update(score(r["dimensions"], TRUTH[dataset]))
    row.update(score_values(r["values"], TRUTH_VALUES[dataset]))
    return row, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--config", choices=CONFIGS)
    args = ap.parse_args()

    configs = [args.config] if args.config else CONFIGS
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / "results.csv"

    total = len(configs) * len(TRUTH) * args.reps
    print(f"{total} runs\n")
    rows, n, t0 = [], 0, time.time()

    for config in configs:
        for dataset in TRUTH:
            for rep in range(args.reps):
                n += 1
                print(f"[{n:>2}/{total}] {config:<10}{dataset:<11}rep {rep}  ",
                      end="", flush=True)

                row, r = one_run(config, dataset, rep)
                rows.append(row)

                if row["error"]:
                    print(f"ERROR {row['error'][:50]}")
                else:
                    print(f"{row['turns']:>2}t {row['tool_calls']:>2}c  "
                          f"{row['declared'] or '(none)':<32} "
                          f"{row['seconds']:>5.0f}s")
                    (ROOT / "runs" / f"{config}_{dataset}_{rep}.json").write_text(
                        json.dumps({"config": config, "dataset": dataset,
                                    "rep": rep, "metrics": row,
                                    "calls": r["calls"], "report": r["report"],
                                    "transcript": r["transcript"]}, indent=2))

                # rewritten each run, so an interruption loses nothing
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    for rr in rows:
                        w.writerow({k: rr.get(k, "") for k in FIELDS})

    # ---------------- summary ----------------
    ok = [r for r in rows if not r["error"]]
    print(f"\n{len(ok)}/{len(rows)} completed in {(time.time()-t0)/60:.1f} min, "
          f"{sum(r['input_tokens'] for r in ok):,} input tokens")
    print(f"\n{'config':<11}{'dataset':<11}{'exact':>7}{'correct':>9}"
          f"{'values':>8}{'inter':>7}{'meas':>6}"
          f"{'spurious':>10}{'turns':>7}{'filtered':>10}")
    for config in configs:
        for dataset in TRUTH:
            g = [r for r in ok if r["config"] == config
                 and r["dataset"] == dataset and r["exact"] != ""]
            if not g:
                continue
            avg = lambda k: sum(r[k] for r in g) / len(g)
            vals = [r for r in g if r["n_values_correct"] != ""]
            vstr = (f"{sum(r['n_values_correct'] for r in vals)/len(vals):>8.1f}"
                    if vals else f"{'-':>8}")
            ni = sum(r["structure"] == "interaction" for r in g)
            nm = sum(r["evidence"] == "measured" for r in g)
            print(f"{config:<11}{dataset:<11}"
                  f"{sum(r['exact'] for r in g):>4}/{len(g):<2}"
                  f"{avg('n_correct'):>9.1f}{vstr}"
                  f"{ni:>4}/{len(g):<2}{nm:>3}/{len(g):<2}"
                  f"{avg('n_spurious'):>10.1f}"
                  f"{avg('turns'):>7.1f}{avg('filtered_calls'):>10.1f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()