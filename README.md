# Tool Schema Design and Root-Cause Discovery in Post-Silicon ATE Data

Companion code for *Does Tool Schema Design Bound the Root-Cause Discovery Rate of a Tool-Calling LLM Agent on Post-Silicon ATE Data?*

**V. N. M. Konduri**
Department of Electrical and Computer Engineering
The University of Texas at Austin

---

## Research Question

The central question investigated by this repository is:

> **Does the schema of the analysis tools place a measurable bound on an LLM agent's ability to discover the root-cause structure of post-silicon ATE excursions?**

The experimental design separates two mechanisms:

**Coverage:**
Can the agent observe the relevant dimensions at all?

**Composition:**
Can the agent combine observations into the specific intersection where the failure is concentrated?

The resulting 40-run matrix measures how these properties affect **correct discovery, false discovery, missed dimensions, interaction identification, evidence quality, and investigation efficiency**.

---

## Overview

A fabless semiconductor company receives test data from overseas assembly and test partners rather than physical silicon. When a production lot's yield falls below its statistical limit, an engineer must diagnose the cause from the datalog alone by choosing which dimension to group along next:

`LOT_ID` · `WAFER_ID` · `SITE_NUM` · `TEMP_C` · `VDD` · `SPATIAL`

A failure may be visible only after the data is grouped along the correct dimension, or only where several dimensions coincide.

This repository implements that investigation as a **tool-calling LLM agent** and measures how much root-cause discovery survives when analysis tools are withheld.

The **model, prompt, data, and iteration limit are identical across every run**. The only experimental variable is which query functions the agent is shown and whether those functions accept row filters.

### Experimental Factors

Two properties of the tool schema are varied independently:

| Property        | Definition                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Coverage**    | How many query functions are exposed, determining which dimensions can be observed                                                         |
| **Composition** | Whether filters are exposed, determining whether one query result can constrain a subsequent query to a specific combination of conditions |

This produces four tool configurations:

|                 | **No Filters** | **Filters** |
| --------------- | -------------- | ----------- |
| **2 Functions** | `minimal`      | `minimal_f` |
| **6 Functions** | `reduced`      | `full`      |

The `reduced` and `full` configurations expose the same six query functions and differ only in whether filtering is available, isolating **composition** from tool-count effects.

---

## Injected Condition

The defect models a **partial ozone-valve restriction during atomic layer deposition of the HfO₂ gate dielectric**.

Reduced oxidant leaves oxygen vacancies, which:

* increase threshold voltage,
* reduce effective mobility,
* lower drive current by approximately **12%**, and
* increase propagation delay by approximately **14%**.

Film thickness remains unchanged, so in-line thickness metrology does not flag the condition.

The affected die pass at most operating corners. They fail only where elevated temperature has removed absolute timing margin and a low supply rail has amplified the threshold shift.

### Ground-Truth Condition

The condition is confined to the intersection of four dimensions:

| Dimension | Injected Value                                  |
| --------- | ----------------------------------------------- |
| `LOT_ID`  | `L47`, `L48` — the two most recent lots         |
| `SPATIAL` | Upper-left perimeter, NW octant at outer radius |
| `TEMP_C`  | `85 °C`                                         |
| `VDD`     | `0.75 V`                                        |

Four of ten tests are affected:

`core_fmax` · `vmin_core` · `pll_lock` · `sram_bist`

All four are drive- or timing-limited.

Both leakage tests are deliberately unaffected. Leakage degrades at high supply voltage, whereas this condition appears at low voltage. This opposite voltage dependence provides a distinguishing feature between the two mechanisms.

---

## Detectability

The condition occupies **339 of 112,500 rows (0.30%)**, spread across **199 of 3,750 die**.

Every single-dimension view is diluted:

| View                 |   Affected |   Baseline |
| -------------------- | ---------: | ---------: |
| By lot               | 0.66–0.69% | 0.14–0.16% |
| By temperature       |      0.81% | 0.11–0.17% |
| By voltage           |      0.78% | 0.14–0.16% |
| By wafer region      |      1.47% |      0.14% |
| **All four crossed** |  **72.0%** |  **0.14%** |

Each dimension is individually visible, but none is individually conclusive.

An agent that cannot restrict rows before aggregating observes four elevated marginals and must infer their relationship without directly measuring the intersection.

This makes the experiment specifically sensitive to whether the tool schema permits the agent to **compose queries across dimensions**.

---

## Repository Structure

```text
src/
├── gen_data.py       # Seeded synthetic datalog generator
├── triage.sh         # Automated excursion gate
├── analyze.py        # Six query functions over the converted datalog
├── tools.py          # JSON schemas, ablation configurations, and dispatch
├── agent.py          # Reason–act investigation loop
├── run_matrix.py     # Experiment driver and scoring
└── report_pdf.py     # Renders individual runs as reviewable PDFs

data/                 # Generated CSVs (not committed)
runs/                 # Gate reports and per-run transcripts
results/              # results.csv and rendered reports
```

---

## Reproduction

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### Generate the Data

```bash
python src/gen_data.py
```

This writes:

```text
data/clean.csv
data/defect_4d.csv
```

The generator is seeded, so both datasets reproduce exactly. The CSVs are intentionally not committed to the repository.

### Run the Excursion Gate

```bash
bash src/triage.sh data/defect_4d.csv
```

### Run the Experiment Matrix

```bash
python src/run_matrix.py
```

This executes all **40 runs** and writes:

```text
results/results.csv
```

### Run a Single Configuration

```bash
python src/agent.py data/defect_4d.csv full
```

### Render a Run as a PDF

```bash
python src/report_pdf.py runs/full_defect_4d_0.json
```

---

## Pipeline

### 1. Generation

Five lots × five wafers × 150 die × three temperature insertions × ten parametric tests produce:

**112,500 rows per dataset**

The columns are derived from STDF-style datalog structure.

Two datasets share the same random seed and differ only by the injected condition.

---

### 2. Excursion Gate

`triage.sh` rolls measurements up to the die level and compares the resulting yield against a statistical yield limit following the approach in **AEC-Q002**.

| Dataset         | Parametric Yield | Disposition |
| --------------- | ---------------: | ----------- |
| Control         |           95.79% | Release     |
| Injected defect |           91.60% | Hold        |

The gate establishes **whether an investigation is required**. It does not attempt to identify the cause.

The control dataset is nevertheless passed to the agent deliberately so that false root-cause reporting can be measured.

---

### 3. Investigation

The agent receives:

* the datalog path,
* the excursion-gate report, and
* the tool schemas for its assigned configuration.

The agent then iterates through a reason–act loop:

```text
Reason
  ↓
Select query
  ↓
Harness validates request
  ↓
Query executes locally
  ↓
Compact result returned
  ↓
Result appended to conversation
  ↓
Repeat
```

All data computation occurs in `analyze.py`.

The model **never receives the raw datalog**. It sees only compact summaries returned by the available query functions.

The loop terminates when the agent submits a final report or reaches **16 iterations**.

---

### 4. Reporting

Each run produces four machine-readable fields followed by a structured report.

The report identifies:

1. the dimensions held responsible,
2. the values associated with those dimensions,
3. whether the dimensions form an interaction or act independently, and
4. whether that relationship was measured directly or inferred from separate results.

---

## Tool Configurations

The experiment uses four configurations:

| Configuration | Functions | Filters | Purpose                           |
| ------------- | --------: | ------: | --------------------------------- |
| `minimal`     |         2 |      No | Reduced coverage, no composition  |
| `minimal_f`   |         2 |     Yes | Reduced coverage with composition |
| `reduced`     |         6 |      No | Full coverage, no composition     |
| `full`        |         6 |     Yes | Full coverage with composition    |

The `reduced` and `full` configurations expose an identical set of six functions. Their only difference is whether the `filters` property appears in the schema.

This isolates the effect of **query composition** from the effect of **tool coverage**.

Filters are removed both from the schema and from dispatch in configurations that do not permit them. A request for an argument that the model was never shown therefore cannot take effect.

### Deliberately Non-Compositional Queries

Two functions return marginal views only:

* `get_fail_rate_by_condition` returns temperature and voltage separately rather than their joint grid.
* `get_yield_by_region` returns radial bands and octants separately rather than their joint grid.

Returning crossed grids would give every configuration direct access to a two-way intersection, undermining the intended composition ablation.

---

## Experiment Design

The full experiment contains:

**4 configurations × 2 datasets × 5 repeats = 40 runs**

Five repeats are used because the investigation loop is stochastic. The model selects each subsequent query from the accumulated conversation, so identical experimental conditions can produce different investigation paths.

### Ground Truth

For the injected dataset, ground truth consists of:

* the four affected dimensions,
* their corresponding values, and
* `interaction`.

For the control dataset, all ground-truth fields are empty.

`WAFER_ID` and `SITE_NUM` correspond to no injected condition and act as distractor dimensions. Declaring either therefore incurs an error.

---

## Scoring

Scoring is automatic.

`results/results.csv` records, for every run:

* dimensions declared,
* correct dimensions,
* incorrect dimensions,
* missed dimensions,
* values correctly identified,
* interaction structure claimed,
* evidence type claimed,
* iterations used,
* queries used,
* number of filtered queries,
* whether the agent concluded autonomously, and
* token usage.

The control dataset allows the experiment to measure **false root-cause discovery**—the rate at which an agent reports a cause when no injected cause exists.

---

## Scope and Limitations

The datalog is simulated after conversion from STDF; no binary STDF records are parsed.

`TEMP_C` and `VDD` are represented as explicit columns. Production datalogs may instead encode test conditions within test names or across separate per-insertion files.

The **95.8% baseline** represents parametric yield only. Gross functional failures, which often dominate wafer-sort loss, are not modeled.

The agent has access only to the datalog and the excursion-gate output. It has no process, equipment, maintenance, or manufacturing-history records.

Therefore, the agent can **localize a failure within the test data but cannot establish its physical process cause**. Every final report states this limitation.

---

## Example Output

[`results/full_defect_4d.pdf`](results/full_defect_4d.pdf) is a report
produced by a single `full` run on the injected dataset. It is included
as an illustration of what the pipeline delivers to a human reviewer.

The run made **25 queries across 12 iterations**, 18 of them filtered,
and used all six available functions. It declared all four injected
dimensions with their correct values.

The report is worth reading for the query appendix rather than the
prose. Queries 12 through 14 show the agent narrowing to the affected
population one condition at a time, then holding that population fixed
while varying the stress conditions:
