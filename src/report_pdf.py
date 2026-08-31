"""
report_pdf.py — render an agent run as a formatted PDF for human review.

The engineer who picks up a flagged lot receives this: run provenance at
the top, the agent's structured findings, and an appendix listing every
query it made so the reasoning can be audited.

Usage:
    python src/report_pdf.py runs/full_defect_4d_0.json
    python src/report_pdf.py runs/*.json
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle,
                                HRFlowable, KeepTogether)

ROOT = Path(__file__).parent.parent

INK = colors.HexColor("#1a1a1a")
MUT = colors.HexColor("#666666")
ACC = colors.HexColor("#1f5fa8")
BAD = colors.HexColor("#b3261e")
RULE = colors.HexColor("#d4d7dd")

SECTIONS = ["FINDINGS", "RULED OUT", "ASSESSMENT", "RECOMMENDED NEXT STEP"]


# ----------------------------------------------------------------------
# styles
# ----------------------------------------------------------------------

def _styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=ss["Normal"], fontName="Helvetica-Bold",
                                fontSize=15, leading=18, textColor=INK, spaceAfter=2)
    s["sub"] = ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                              fontSize=8.5, leading=11, textColor=MUT, spaceAfter=10)
    s["h"] = ParagraphStyle("h", parent=ss["Normal"], fontName="Helvetica-Bold",
                            fontSize=10, leading=13, textColor=ACC,
                            spaceBefore=13, spaceAfter=5)
    s["body"] = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=9, leading=13.2, textColor=INK,
                               alignment=TA_LEFT, spaceAfter=5)
    s["item"] = ParagraphStyle("i", parent=s["body"], leftIndent=14,
                               bulletIndent=3, spaceAfter=4)
    s["verdict"] = ParagraphStyle("v", parent=ss["Normal"], fontName="Helvetica-Bold",
                                  fontSize=10, leading=13, textColor=INK)
    s["mono"] = ParagraphStyle("m", parent=ss["Normal"], fontName="Courier",
                               fontSize=7.2, leading=9.4, textColor=INK)
    s["foot"] = ParagraphStyle("f", parent=ss["Normal"], fontName="Helvetica-Oblique",
                               fontSize=7.5, leading=10, textColor=MUT)
    return s


def _esc(t):
    """Escape for reportlab markup, then convert **bold** to a bold tag."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


# ----------------------------------------------------------------------
# report parsing
# ----------------------------------------------------------------------

def _split_sections(report):
    """
    Split the agent's report on its required headings.

    Anything before the first heading is preamble the model emitted while
    still reasoning; it is kept but marked, since it is not part of the
    structured output.
    """
    pat = re.compile(r"^\s*#*\s*(" + "|".join(SECTIONS) + r")\s*$",
                     re.MULTILINE | re.IGNORECASE)
    marks = list(pat.finditer(report or ""))
    if not marks:
        return {"_preamble": (report or "").strip()}

    out = {}
    pre = report[:marks[0].start()].strip()
    if pre:
        out["_preamble"] = pre
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(report)
        out[m.group(1).upper()] = report[m.end():end].strip()
    return out


def _flow(text, st):
    """Turn a section body into paragraphs, preserving list structure."""
    out = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or set(line) <= {"-", "ـ", "—", "#"}:
            continue
        line = re.sub(r"^#+\s*", "", line)
        m = re.match(r"^(\d+)[.)]\s+(.*)", line)
        if m:
            out.append(Paragraph(_esc(m.group(2)), st["item"],
                                 bulletText=f"{m.group(1)}."))
            continue
        if re.match(r"^[-*\u2022]\s+", line):
            out.append(Paragraph(_esc(re.sub(r"^[-*\u2022]\s+", "", line)),
                                 st["item"], bulletText="\u2022"))
            continue
        out.append(Paragraph(_esc(line), st["body"]))
    return out


# ----------------------------------------------------------------------
# page furniture
# ----------------------------------------------------------------------

def _decorate(canvas, doc):
    canvas.saveState()
    w, h = LETTER
    canvas.setFillColor(MUT)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(0.85 * inch, 0.55 * inch,
                      "Automated yield-excursion triage \u2014 for engineering review")
    canvas.drawRightString(w - 0.85 * inch, 0.55 * inch, f"page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(0.85 * inch, 0.72 * inch, w - 0.85 * inch, 0.72 * inch)
    canvas.restoreState()


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------

def build(run_json, out_pdf=None):
    rec = json.loads(Path(run_json).read_text())
    m = rec.get("metrics", {})
    st = _styles()

    out_pdf = Path(out_pdf) if out_pdf else \
        Path(run_json).with_suffix("").with_name(Path(run_json).stem + ".pdf")

    doc = BaseDocTemplate(str(out_pdf), pagesize=LETTER,
                          leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                          topMargin=0.8 * inch, bottomMargin=0.85 * inch,
                          title=f"Triage report \u2014 {rec.get('dataset','')}",
                          author="automated triage")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_decorate)])

    story = []

    # ---- header ----
    story.append(Paragraph("Yield Excursion \u2014 Root-Cause Triage Report", st["title"]))
    story.append(Paragraph(
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} "
        f"&nbsp;\u00b7&nbsp; datalog <b>{rec.get('dataset','')}</b>", st["sub"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=RULE,
                            spaceBefore=0, spaceAfter=10))

    # ---- provenance ----
    declared = m.get("declared") or "\u2014"
    meta = [
        ["tool configuration", rec.get("config", "")],
        ["iterations", f"{m.get('turns','')}  ({m.get('stop_reason','')})"],
        ["queries issued", f"{m.get('tool_calls','')}  "
                           f"({m.get('filtered_calls',0)} filtered, "
                           f"{m.get('distinct_tools','')} distinct functions)"],
        ["dimensions reported", declared.replace("|", ", ")],
    ]
    t = Table(meta, colWidths=[1.55 * inch, doc.width - 1.55 * inch])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica", 8.5),
        ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUT),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("TEXTCOLOR", (1, 3), (1, 3), BAD if declared not in ("", "\u2014", "NO_ANSWER_LINE") else MUT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This report is produced from the converted test datalog only. "
        "No process, equipment, or maintenance records were available to "
        "the analysis. Findings localize a failure within the data; they "
        "do not establish a physical cause. Disposition remains an "
        "engineering judgement.", st["foot"]))

    # ---- body ----
    parts = _split_sections(rec.get("report", ""))
    if "_preamble" in parts and len(parts) == 1:
        story.append(Paragraph("Report", st["h"]))
        story += _flow(parts["_preamble"], st)
    else:
        for name in SECTIONS:
            if name not in parts:
                continue
            block = [Paragraph(name.title(), st["h"])] + _flow(parts[name], st)
            story.append(KeepTogether(block[:2]))
            story += block[2:]

    # ---- appendix ----
    calls = rec.get("calls", [])
    if calls:
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                spaceBefore=4, spaceAfter=8))
        story.append(Paragraph("Appendix \u2014 Queries Issued", st["h"]))
        story.append(Paragraph(
            "Every query made during the investigation, in order. Included "
            "so that each finding above can be traced to the data that "
            "produced it.", st["foot"]))
        story.append(Spacer(1, 5))

        rows = [["#", "iter", "query", "restricted to"]]
        for i, c in enumerate(calls, 1):
            args = dict(c.get("args") or {})
            filt = args.pop("filters", None)
            extra = ", ".join(f"{k}={v}" for k, v in args.items())
            name = c["tool"] + (f" ({extra})" if extra else "")
            rows.append([
                str(i), str(c.get("turn", "")),
                Paragraph(_esc(name), st["mono"]),
                Paragraph(_esc(", ".join(f"{k}={v}" for k, v in filt.items()))
                          if filt else "\u2014", st["mono"]),
            ])
        at = Table(rows, colWidths=[0.28 * inch, 0.36 * inch,
                                    2.85 * inch, doc.width - 3.49 * inch],
                   repeatRows=1)
        at.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
            ("FONT", (0, 1), (1, -1), "Courier", 7.2),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUT),
            ("TEXTCOLOR", (0, 1), (1, -1), MUT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(at)

    doc.build(story)
    return out_pdf


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(p) for p in sorted((ROOT / "runs").glob("*.json"))]
    if not paths:
        print("no run json files found in runs/")
        sys.exit(1)
    for p in paths:
        print("wrote", build(p))