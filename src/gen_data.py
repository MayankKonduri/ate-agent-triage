"""
gen_data.py — Synthetic post-silicon ATE datalog generator.

Simulates STDF datalog data AFTER conversion to a structured table.
Writes four datasets to data/:

    baseline_allpass.csv  limits at 6 sigma, nothing fails    -> validation only
    clean.csv             no injected defect                  -> NONE     (depth 0)
    defect_lot.csv        lot Fmax/Vmin shift, yield-visible  -> LOT_ID   (depth 1)
    defect_lot_subtle.csv same cause, pre-yield-loss           -> LOT_ID   (depth 2)
    defect_site.csv       test-site drift over time           -> SITE_NUM (depth 1)
    defect_temp.csv       cold-corner Vmin_sram shift         -> TEMP_C   (depth 2)
    defect_edge.csv       edge-ring IO leakage signature      -> SPATIAL  (depth 2)
    defect_edge_temp.csv  edge x hot interaction on leakage   -> SPATIAL x TEMP_C (depth 3)

Seeded, so the datasets are reproducible from this file alone.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SEED = 42

LOT_IDS = ["L44", "L45", "L46", "L47", "L48"]
N_WAFERS = 5          # wafers per lot
N_DIE = 150           # die per wafer
N_SITES = 8           # parallel test sites on the tester

# Each part is tested at three insertions (cold / room / hot).
INSERTIONS = [
    ("sort_cold", 0),
    ("sort_room", 25),
    ("sort_hot", 85),
]

VDD_LEVELS = [0.75, 0.80, 0.85]

# 5 lots x 5 wafers x 150 die x 3 insertions x 10 tests = 112,500 rows
OUT_DIR = Path(__file__).parent.parent / "data"


# ----------------------------------------------------------------------
# Test definitions
#
# Every test gets both limits. The inactive side is parked at +/- 6 sigma
# so it never trips, which keeps the schema uniform without needing
# null handling. Active limits sit at ~3 sigma -> ~0.2% fail per test,
# ~2% part-level baseline across 10 tests.
# ----------------------------------------------------------------------

TESTS = [
    # test_num, test_txt,        mean,   sigma,  lo,     hi,     units
    (1000, "core_fmax",          3.850,  0.080,  3.610,  4.330,  "GHz"),
    (1010, "vmin_core",          0.720,  0.015,  0.630,  0.765,  "V"),
    (1020, "vmin_sram",          0.750,  0.018,  0.642,  0.804,  "V"),
    (2000, "idd_static",         0.450,  0.040,  0.210,  0.570,  "A"),
    (2010, "idd_dynamic",       12.500,  0.800,  7.700, 14.900,  "A"),
    (3000, "sram_bist",          8.000,  1.200,  0.800, 11.600,  "repairs"),
    (3010, "scan_stuck_at",     99.200,  0.250, 98.450,100.700,  "pct"),
    (4000, "pll_lock",          45.000,  4.000, 21.000, 57.000,  "us"),
    (4010, "io_leakage",         1.800,  0.300,  0.000,  2.700,  "uA"),
    (5000, "thermal_diode",      0.500,  0.150, -0.400,  0.950,  "degC"),
]

TEST_COLS = ["TEST_NUM", "TEST_TXT", "_MEAN", "_SIGMA",
             "LO_LIMIT", "HI_LIMIT", "UNITS"]


# ----------------------------------------------------------------------
# Base dataset
# ----------------------------------------------------------------------

def wafer_coords(n_die):
    """Return n_die (x, y) positions arranged in a rough wafer circle."""
    side = int(np.ceil(np.sqrt(n_die * 4 / np.pi)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    xs, ys = xs.ravel(), ys.ravel()
    c = (side - 1) / 2
    r = np.sqrt((xs - c) ** 2 + (ys - c) ** 2)
    keep = np.argsort(r)[:n_die]
    return xs[keep], ys[keep]


def build_base(rng):
    """Build the full parts x insertions x tests table with clean results."""
    x_coord, y_coord = wafer_coords(N_DIE)

    # ---- parts ----
    parts = []
    part_counter = 0
    for lot in LOT_IDS:
        for w in range(1, N_WAFERS + 1):
            for d in range(N_DIE):
                part_counter += 1
                parts.append((
                    lot,
                    f"W{w:02d}",
                    f"P{part_counter:06d}",
                    int(x_coord[d]),
                    int(y_coord[d]),
                    (d % N_SITES) + 1,       # multisite probing
                ))
    parts = pd.DataFrame(
        parts,
        columns=["LOT_ID", "WAFER_ID", "PART_ID",
                 "X_COORD", "Y_COORD", "SITE_NUM"],
    )

    # ---- cross with insertions ----
    ins = pd.DataFrame(INSERTIONS, columns=["INSERTION", "TEMP_C"])
    df = parts.merge(ins, how="cross")

    # ---- cross with tests ----
    tests = pd.DataFrame(TESTS, columns=TEST_COLS)
    df = df.merge(tests, how="cross")

    # ---- per-row conditions ----
    n = len(df)
    df["VDD"] = rng.choice(VDD_LEVELS, size=n)

    # Touchdown order: drives the site-drift injection and TIMESTAMP.
    df = df.sort_values(["LOT_ID", "WAFER_ID", "PART_ID"]).reset_index(drop=True)
    df["_PROGRESS"] = np.arange(n) / n

    df["TIMESTAMP"] = pd.Timestamp("2026-06-01") + pd.to_timedelta(
        np.arange(n) * 0.4, unit="s"
    )

    # ---- clean measurements ----
    df["RESULT"] = rng.normal(df["_MEAN"].values, df["_SIGMA"].values)

    return df


# ----------------------------------------------------------------------
# Defect injections
#
# Each shifts the mean of one or more tests along exactly one dimension.
# The magnitudes below are starting points — verify and tune them in
# Step 2 before building anything on top.
# ----------------------------------------------------------------------

def widen_limits(df):
    """
    Validation baseline: park every limit at +/-6 sigma so no measurement
    can fail by construction.

    This is not a control condition — it is a correctness check. If any
    part fails here, the pass/fail or yield logic is wrong, because the
    probability of a 6 sigma excursion across 112,500 draws is ~2e-4.

    The realistic control is clean.csv, which keeps 3 sigma limits and so
    carries a natural ~4% part-level failure rate.
    """
    df = df.copy()
    df["LO_LIMIT"] = (df["_MEAN"] - 6 * df["_SIGMA"]).round(4)
    df["HI_LIMIT"] = (df["_MEAN"] + 6 * df["_SIGMA"]).round(4)
    return df


def inject_lot(df):
    """
    Lot-level material issue in L47.

    Slow silicon needs more voltage, so the Fmax shift is paired with a
    correlated Vmin shift. Independent single-test defects look artificial.
    Visible in an aggregate yield-by-lot view.
    """
    m = df["LOT_ID"] == "L47"

    fmax = m & (df["TEST_TXT"] == "core_fmax")
    df.loc[fmax, "RESULT"] -= 0.090         # GHz slower

    vmin = m & (df["TEST_TXT"] == "vmin_core")
    df.loc[vmin, "RESULT"] += 0.014         # V higher

    return df


def inject_lot_subtle(df):
    """
    Same root cause as inject_lot, but caught earlier — before the shift
    has eaten much yield.

    L47's yield sits only ~2 points below its neighbours, inside the
    normal lot-to-lot spread, so a yield-by-lot view gives a weak hint at
    best. Confirming it requires comparing the core_fmax distribution
    against a reference lot, where a ~0.5 sigma mean shift is unambiguous.

    Depth 2: yield by lot -> distribution comparison.
    """
    m = df["LOT_ID"] == "L47"

    fmax = m & (df["TEST_TXT"] == "core_fmax")
    df.loc[fmax, "RESULT"] -= 0.040

    vmin = m & (df["TEST_TXT"] == "vmin_core")
    df.loc[vmin, "RESULT"] += 0.006

    return df


def inject_edge(df):
    """
    Edge-ring signature: elevated IO leakage on die near the wafer edge.

    A classic process signature — film thickness and etch uniformity
    degrade toward the wafer edge, so the outermost die see systematically
    worse leakage.

    This defect is uniform across every non-spatial dimension. Yield by
    lot, wafer, site and test condition are all flat. The failure Pareto
    identifies io_leakage as the failing test but gives no clue where the
    failures are concentrated. It is only localisable by die coordinate,
    which makes it unreachable without a spatial tool regardless of model
    capability.

    Depth 2: failure Pareto -> spatial distribution.
    """
    cx = df["X_COORD"].max() / 2
    cy = df["Y_COORD"].max() / 2
    r = np.sqrt((df["X_COORD"] - cx) ** 2 + (df["Y_COORD"] - cy) ** 2)
    r_norm = r / r.max()

    m = (r_norm > 0.82) & (df["TEST_TXT"] == "io_leakage")
    df.loc[m, "RESULT"] += 0.45              # uA higher at the edge

    return df


def inject_edge_temp(df):
    """
    Edge x temperature interaction: IO leakage fails only on edge die at
    the hot insertion.

    Physical story: junction leakage is strongly temperature-activated,
    and edge die have degraded isolation from film and etch non-uniformity
    at the wafer perimeter. Neither condition alone pushes the part over
    limit; together they do.

    This is a true interaction, not two additive effects:

        edge     @ 85C  ->  12.2% fail
        edge     @  0C  ->   0.2% fail   (edge alone is fine)
        interior @ 85C  ->   0.0% fail   (hot alone is fine)

    Each marginal view gives a real but incomplete hint. Grouping by
    position says "the edge is bad"; grouping by temperature says "hot is
    bad". Both are wrong as stated. Only crossing the two dimensions
    shows the failure is confined to their intersection, which means the
    agent must carry a result from one grouping into a second.

    Depth 3: failure Pareto -> spatial -> spatial x temperature.
    """
    cx = df["X_COORD"].max() / 2
    cy = df["Y_COORD"].max() / 2
    r = np.sqrt((df["X_COORD"] - cx) ** 2 + (df["Y_COORD"] - cy) ** 2)
    r_norm = r / r.max()

    m = (r_norm > 0.80) & (df["TEMP_C"] == 85) & (df["TEST_TXT"] == "io_leakage")
    df.loc[m, "RESULT"] += 0.55

    return df


def inject_quadrant_hot(df):
    """
    Localized process asymmetry: upper-left wafer edge sector, hot only.

    Physical story: asymmetric gas flow or edge-bead misalignment in a
    single-wafer chamber leaves one sector of the wafer perimeter with
    thinner oxide and wider dopant spread. The resulting threshold and
    leakage degradation is latent at cold and room temperature and only
    crosses limits at 85C, where leakage is exponentially activated and
    drive current is weakest.

    Designed so the failure Pareto is UNINFORMATIVE. Six temperature-
    sensitive parameters degrade together at similar rates (0.8-1.0%
    each). No single test dominates, which is itself the diagnostic
    signal: a test-specific mechanism would produce one tall bar, whereas
    several unrelated parameters moving together points at material or
    process, not at test content or a mis-set limit.

    Detection chain:
        1. Pareto by test        -> six tests comparable, none dominant
        2. Fail rate by temp     -> 0.11% / 0.17% / 1.58%  (hot 10x)
        3. Wafer map at 85C      -> upper-left perimeter lights up
        4. Isolate that sector   -> 61% of parts fail vs 4% elsewhere

    Yield: 86.3% overall, 38.7% inside the sector, 95.9% outside.

    Answer: SPATIAL x TEMP_C  (2 dimensions, 4 steps)
    """
    cx = df["X_COORD"].max() / 2
    cy = df["Y_COORD"].max() / 2
    dx, dy = df["X_COORD"] - cx, df["Y_COORD"] - cy

    r_norm = np.sqrt(dx ** 2 + dy ** 2)
    r_norm = r_norm / r_norm.max()
    angle = np.degrees(np.arctan2(dy, dx)) % 360        # 0 = right, 90 = up

    sector = (r_norm > 0.55) & (angle >= 100) & (angle <= 195)   # 16.7% of die

    # Shift in sigma-multiples, so each test moves by a comparable amount
    # relative to its own distribution. Sign follows the physics: Fmax
    # drops, everything else rises.
    # Sensitivity ordering follows the physics: junction leakage is the
    # most strongly temperature-activated, static Idd next, then drive
    # current (Fmax), then threshold-driven Vmin, then PLL timing. The
    # spread matters — identical shifts across tests would be an
    # artefact, not a mechanism.
    affected = {
        "io_leakage":   1.85,
        "idd_static":   1.60,
        "core_fmax":   -1.35,
        "idd_dynamic":  1.15,
        "vmin_core":    1.00,
        "pll_lock":     0.85,
    }
    scale = 1.35

    for test, k in affected.items():
        m = sector & (df["TEMP_C"] == 85) & (df["TEST_TXT"] == test)
        df.loc[m, "RESULT"] += k * scale * df.loc[m, "_SIGMA"]

    return df


def inject_site(df):
    """
    Test site 3 drifting out of calibration over the course of the run.

    Site 3 is 1/8 of parts and the drift ramps from 0, so the aggregate
    yield barely moves. Obvious only when grouped by SITE_NUM.
    """
    m = (df["SITE_NUM"] == 3) & (df["TEST_TXT"] == "core_fmax")
    df.loc[m, "RESULT"] -= 0.120 * df.loc[m, "_PROGRESS"]
    return df


def inject_temp(df):
    """
    SRAM Vmin marginality that only appears at the cold corner.

    TEMP_C == 0 is 1/3 of insertions, so the aggregate signal is diluted.
    Requires splitting by test condition to see.
    """
    m = (df["TEMP_C"] == 0) & (df["TEST_TXT"] == "vmin_sram")
    df.loc[m, "RESULT"] += 0.019            # V higher at cold
    return df


# ----------------------------------------------------------------------
# Pass/fail and binning
# ----------------------------------------------------------------------

def finalize(df):
    """Apply limits, assign bins, drop internal columns."""
    df = df.copy()

    in_limits = (df["RESULT"] >= df["LO_LIMIT"]) & (df["RESULT"] <= df["HI_LIMIT"])
    df["PASS_FAIL"] = np.where(in_limits, "P", "F")

    # SOFT_BIN: 1 = pass, otherwise the test number that failed.
    df["SOFT_BIN"] = np.where(in_limits, 1, df["TEST_NUM"])

    # HARD_BIN: 1 = pass, 2 = parametric fail, 3 = functional fail.
    functional = df["TEST_NUM"] >= 3000
    df["HARD_BIN"] = np.where(in_limits, 1, np.where(functional, 3, 2))

    cols = [
        "LOT_ID", "WAFER_ID", "PART_ID", "X_COORD", "Y_COORD", "SITE_NUM",
        "TEST_NUM", "TEST_TXT", "RESULT", "LO_LIMIT", "HI_LIMIT", "UNITS",
        "PASS_FAIL", "HARD_BIN", "SOFT_BIN", "TEMP_C", "VDD",
        "INSERTION", "TIMESTAMP",
    ]
    df["RESULT"] = df["RESULT"].round(4)
    return df[cols]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

DATASETS = {
    "clean":           None,               # control  -> expected answer: NONE
    "defect_quad_hot": inject_quadrant_hot, # -> SPATIAL x TEMP_C

    # Additional scenarios are implemented above and can be re-enabled by
    # adding them here. Left out of the active set for now.
    #   "baseline_allpass":  widen_limits,      # harness validation, 100% yield
    #   "defect_lot":        inject_lot,        # LOT_ID,   1 dim, 1 step
    #   "defect_lot_subtle": inject_lot_subtle, # LOT_ID,   1 dim, 2 steps
    #   "defect_site":       inject_site,       # SITE_NUM, 1 dim, 1 step
    #   "defect_temp":       inject_temp,       # TEMP_C,   1 dim, 2 steps
    #   "defect_edge":       inject_edge,       # SPATIAL,  1 dim, 2 steps
    #   "defect_edge_temp":  inject_edge_temp,  # SPATIAL x TEMP_C
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, injector in DATASETS.items():
        rng = np.random.default_rng(SEED)      # same base data every time
        df = build_base(rng)

        if injector is not None:
            df = injector(df)

        out = finalize(df)
        path = OUT_DIR / f"{name}.csv"
        out.to_csv(path, index=False)

        # quick sanity line
        part_pass = out.groupby("PART_ID")["PASS_FAIL"].apply(lambda s: (s == "F").sum() == 0)
        yld = 100 * part_pass.mean()
        print(f"{name:14s} rows={len(out):>7,}  parts={len(part_pass):>6,}  yield={yld:5.1f}%")


if __name__ == "__main__":
    main()