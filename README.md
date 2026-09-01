# Tool Schema Design and Root-Cause Discovery in Post-Silicon ATE Data

Companion code for *Does Tool Schema Design Bound the Root-Cause Discovery
Rate of a Tool-Calling LLM Agent on Post-Silicon ATE Data?*

V. N. M. Konduri, Dept. of Electrical and Computer Engineering,
The University of Texas at Austin.

---

## Summary

A fabless semiconductor company receives test data from overseas assembly
and test partners rather than physical silicon. When a production lot's
yield falls below its statistical limit, an engineer must diagnose the
cause from that data alone, by choosing which dimension to group along
next — lot, wafer, test site, temperature, supply voltage, or die
position. A failure may be visible only after the data is grouped along
the correct dimension, or only where several dimensions coincide.

This repository implements that investigation as a tool-calling LLM agent
and measures how much of it survives when analysis tools are withheld.
The model, prompt, data, and iteration limit are identical across every
run. The only variable is which query functions the agent is shown and
whether those functions accept row filters.

Two properties of the tool schema are varied independently:

**Coverage** — how many query functions are exposed, and therefore which
dimensions can be observed at all.

**Composition** — whether filters are exposed, and therefore whether a
result from one query can restrict a subsequent one to a specific
combination of conditions.

---

## The injected condition

The defect models a partial ozone-valve restriction during atomic layer
deposition of the HfO₂ gate dielectric. Reduced oxidant leaves oxygen
vacancies, which raise threshold voltage and reduce effective mobility,
lowering drive current by roughly 12% and extending propagation delay by
roughly 14%. Film thickness is unchanged, so in-line thickness metrology
does not flag it.

The affected die pass at most operating corners. They fail only where
elevated temperature has removed absolute timing margin and a low supply
rail has amplified the threshold shift.

The condition is confined to the intersection of four dimensions:

| Dimension | Value |
|---|---|
| `LOT_ID` | L47, L48 — the two most recent lots |
| `SPATIAL` | upper-left perimeter, NW octant at outer radius |
| `TEMP_C` | 85 °C |
| `VDD` | 0.75 V |

Four of ten tests are affected — `core_fmax`, `vmin_core`, `pll_lock`,
`sram_bist` — all drive- or timing-limited. Both leakage tests are
deliberately unaffected: leakage degrades at *high* supply voltage while
this condition appears at *low* voltage, and that opposite dependence
distinguishes the two mechanisms.

**Detectability.** The condition occupies 339 of 112,500 rows (0.30%),
spread across 199 of 3,750 die. Every single-dimension view is diluted:

| View | Affected | Baseline |
|---|---|---|
| by lot | 0.66–0.69% | 0.14–0.16% |
| by temperature | 0.81% | 0.11–0.17% |
| by voltage | 0.78% | 0.14–0.16% |
| by wafer region | 1.47% | 0.14% |
| **all four crossed** | **72.0%** | 0.14% |

Each dimension is individually visible and none is individually
conclusive. An agent that cannot restrict rows before aggregating
observes four elevated marginals and must infer their relationship
without measuring it.

---

## Repository layout

```
src/
  gen_data.py      synthetic datalog generator, seeded
  triage.sh        automated excursion gate, writes a disposition report
  analyze.py       six query functions over the converted datalog
  tools.py         JSON schemas, ablation configurations, dispatch
  agent.py         the reason–act investigation loop
  run_matrix.py    experiment driver and scoring
  report_pdf.py    renders one run as a reviewable PDF

data/              generated CSVs (not committed; reproduce from seed)
runs/              gate reports and per-run transcripts
results/           results.csv and rendered reports
```

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

python src/gen_data.py                  # writes data/clean.csv, data/defect_4d.csv
bash src/triage.sh data/defect_4d.csv   # excursion gate
python src/run_matrix.py                # 40 runs, writes results/results.csv
```

`gen_data.py` is seeded, so both datasets reproduce exactly. The CSVs are
not committed; regenerate them.

Single run, with a rendered report:

```bash
python src/agent.py data/defect_4d.csv full
python src/report_pdf.py runs/full_defect_4d_0.json
```

---

## Pipeline

**1. Generation.** Five lots × five wafers × 150 die × three temperature
insertions × ten parametric tests = 112,500 rows per dataset. Columns are
STDF-derived. Two datasets share a random seed and differ only by the
injected condition.

**2. Excursion gate.** `triage.sh` rolls measurements up to die-level
yield and compares against a statistical yield limit, following the
approach in AEC-Q002. The control releases at 95.79%; the injected
dataset is held at 91.60%. The gate establishes *whether* to investigate;
it does not investigate.

**3. Investigation.** The agent receives the datalog path, the gate
report, and the schemas for its configuration, then iterates: it requests
a query, the harness validates and executes it locally, and the result is
appended to a conversation that is retransmitted in full each turn. All
computation happens in `analyze.py`; the model never sees the datalog,
only compact summaries. The loop ends when the agent reports or after
sixteen iterations.

**4. Reporting.** Each run emits four machine-readable lines followed by
a structured report. The lines name the dimensions held responsible, the
values within each, whether those dimensions form an interaction or act
independently, and whether that relationship was measured directly or
inferred from separate results.

---

## Configurations

|  | no filters | filters |
|---|---|---|
| **2 functions** | `minimal` | `minimal_f` |
| **6 functions** | `reduced` | `full` |

`reduced` and `full` expose an identical set of six functions and differ
only in whether the `filters` property appears in the schema, isolating
composition from the confound of tool count.

Filters are removed both from the schema and at dispatch in
configurations that do not permit them, so a request for an argument the
model was never shown cannot take effect.

Two functions deliberately return marginals only. `get_fail_rate_by_condition`
returns temperature and voltage separately rather than their grid;
`get_yield_by_region` returns radial bands and octants separately rather
than their grid. Returning a crossed grid would hand every configuration
a two-way intersection regardless of whether it was permitted to
construct one.

---

## Experiment

4 configurations × 2 datasets × 5 repeats = 40 runs. Repeats are required
because the loop is stochastic: the model selects each query from the
accumulated conversation, so identical conditions may produce different
paths.

Scoring is automatic. Ground truth for the injected dataset is the four
dimensions above, their values, and `interaction`; for the control, all
are empty. `WAFER_ID` and `SITE_NUM` correspond to no injected condition
and act as distractors, so declaring a dimension carries a cost.

The control dataset passes the gate and would never reach an agent in
production. It is run deliberately, since the rate at which a cause is
reported where none exists is otherwise unobservable.

`results/results.csv` records, per run: dimensions declared, correct,
incorrect and missed; values correct; structure and evidence claimed;
iterations and queries used; how many queries applied filters; whether
the run concluded on its own; and token usage.

---

## Notes on scope

The datalog is simulated after conversion from STDF; no binary records
are parsed. `TEMP_C` and `VDD` are represented as explicit columns,
whereas production datalogs typically encode test conditions in test
names or across separate per-insertion files. The 95.8% baseline
represents parametric yield only; gross functional failures, which
usually dominate wafer-sort loss, are not modelled.

The agent has access to the datalog alone, with no process, equipment, or
maintenance records. It can localize a failure within the data but cannot
establish its physical cause, and every report states this.
