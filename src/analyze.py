"""
analyze.py — query functions over a converted ATE datalog.

Each function performs a single aggregation and returns a small
JSON-serialisable dict. Nothing here interprets, ranks, or flags: the
functions report numbers and the caller decides what they mean. Keeping
interpretation out of the tools is what makes the tool-schema ablation
meaningful -- a tool that pre-identified the anomaly would move the
reasoning out of the agent.

Every function accepts an optional `filters` argument restricting the
rows considered BEFORE aggregation. This is the capability the ablation
manipulates. Without it every result is a marginal, averaged over each
dimension not being grouped on.

    get_overall_yield(csv, filters)
    get_failures_by_test(csv, filters)
    get_yield_summary(csv, group_by, filters)
    get_fail_rate_by_condition(csv, test_txt, filters)
    get_yield_by_region(csv, test_txt, filters)
"""

from pathlib import Path
import numpy as np
import pandas as pd

_CACHE = {}

GROUP_FIELDS = ["LOT_ID", "WAFER_ID", "SITE_NUM"]
FILTER_FIELDS = ["LOT_ID", "WAFER_ID", "SITE_NUM", "TEST_TXT",
                 "TEMP_C", "VDD", "INSERTION", "REGION", "OCTANT"]

# radial band edges, normalised 0-1 from wafer centre
_BANDS = [0.0, 0.35, 0.55, 0.75, 1.01]
_BAND_NAMES = ["centre", "mid", "outer", "edge"]


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def _load(csv):
    """Read once and cache. Adds derived RADIUS / ANGLE / REGION columns."""
    key = str(Path(csv).resolve())
    if key in _CACHE:
        return _CACHE[key]

    df = pd.read_csv(csv)

    cx = df["X_COORD"].max() / 2
    cy = df["Y_COORD"].max() / 2
    dx, dy = df["X_COORD"] - cx, df["Y_COORD"] - cy

    r = np.sqrt(dx ** 2 + dy ** 2)
    df["RADIUS"] = (r / r.max()).round(3)
    df["ANGLE"] = np.degrees(np.arctan2(dy, dx)).round(1) % 360

    df["REGION"] = pd.cut(df["RADIUS"], _BANDS,
                          labels=_BAND_NAMES, include_lowest=True)
    # 45-degree octants, named by the compass direction of their centre
    oct_idx = (df["ANGLE"] // 45).astype(int)
    df["OCTANT"] = oct_idx.map(dict(enumerate(
        ["E", "NE", "N", "NW", "W", "SW", "S", "SE"])))

    _CACHE[key] = df
    return df


def _apply(df, filters):
    """
    Restrict rows before aggregating.

    filters is a dict of column -> value or list of values, e.g.
        {"LOT_ID": ["L47", "L48"], "TEMP_C": 85, "VDD": 0.75}
    """
    if not filters:
        return df, {}
    applied = {}
    for col, val in filters.items():
        if col not in df.columns:
            continue
        vals = val if isinstance(val, (list, tuple)) else [val]
        df = df[df[col].isin(vals)]
        applied[col] = vals if len(vals) > 1 else vals[0]
    return df, applied


def _part_yield(df):
    """Percentage of die with no failing measurement."""
    if df.empty:
        return None
    passed = df.groupby("PART_ID")["PASS_FAIL"].apply(lambda s: (s == "F").sum() == 0)
    return float(round(100 * passed.mean(), 2))


def _fail_rate(df):
    """Percentage of measurements outside limits."""
    if df.empty:
        return None
    return float(round(100 * (df["PASS_FAIL"] == "F").mean(), 3))


def _wrap(df, filters, payload):
    """Attach the row count and the filters that were honoured."""
    out = {"rows": int(len(df))}
    if filters:
        out["filters_applied"] = filters
    out.update(payload)
    return out


# ----------------------------------------------------------------------
# 1. overall yield
# ----------------------------------------------------------------------

def get_overall_yield(csv, filters=None):
    """Part-level yield and failure counts for the selected rows."""
    df, f = _apply(_load(csv), filters)
    if df.empty:
        return {"rows": 0, "error": "no rows match those filters"}
    n_parts = df["PART_ID"].nunique()
    y = _part_yield(df)
    return _wrap(df, f, {
        "parts": int(n_parts),
        "parts_failed": int(round(n_parts * (100 - y) / 100)),
        "part_yield_pct": y,
        "measurement_fail_rate_pct": _fail_rate(df),
    })


# ----------------------------------------------------------------------
# 2. failure Pareto by test
# ----------------------------------------------------------------------

def get_failures_by_test(csv, filters=None):
    """Failure count and rate for each test, ranked worst first."""
    df, f = _apply(_load(csv), filters)
    if df.empty:
        return {"rows": 0, "error": "no rows match those filters"}
    g = df.groupby("TEST_TXT")["PASS_FAIL"]
    tbl = pd.DataFrame({
        "runs": g.size(),
        "fails": g.apply(lambda s: (s == "F").sum()),
    })
    tbl["fail_rate_pct"] = (100 * tbl.fails / tbl.runs).round(3)
    tbl = tbl.sort_values("fail_rate_pct", ascending=False)
    return _wrap(df, f, {"tests": [
        {"test": t, "runs": int(r.runs), "fails": int(r.fails),
         "fail_rate_pct": float(r.fail_rate_pct)}
        for t, r in tbl.iterrows()]})


# ----------------------------------------------------------------------
# 3. yield grouped by a categorical field
# ----------------------------------------------------------------------

def get_yield_summary(csv, group_by, filters=None):
    """Part-level yield and measurement fail rate per group."""
    if group_by not in GROUP_FIELDS:
        return {"error": f"group_by must be one of {GROUP_FIELDS}"}
    df, f = _apply(_load(csv), filters)
    if df.empty:
        return {"rows": 0, "error": "no rows match those filters"}
    rows = []
    for key, sub in df.groupby(group_by, observed=True):
        rows.append({
            group_by: key if not isinstance(key, np.generic) else key.item(),
            "parts": int(sub["PART_ID"].nunique()),
            "part_yield_pct": _part_yield(sub),
            "fail_rate_pct": _fail_rate(sub),
        })
    rows.sort(key=lambda r: r["part_yield_pct"])
    return _wrap(df, f, {"group_by": group_by, "groups": rows})


# ----------------------------------------------------------------------
# 4. fail rate across test conditions
# ----------------------------------------------------------------------

def get_fail_rate_by_condition(csv, test_txt=None, filters=None):
    """
    Failure rate split by temperature and by supply voltage.

    Returns the two marginals SEPARATELY and does not return the
    TEMP_C x VDD grid. Crossing the two is composition, and composition
    is the capability the ablation manipulates: returning a crossed grid
    here would hand every configuration a two-way intersection whether
    or not it was permitted to construct one. A configuration with
    filters can obtain any cell by restricting rows first.
    """
    df, f = _apply(_load(csv), filters)
    if test_txt:
        df = df[df["TEST_TXT"] == test_txt]
    if df.empty:
        return {"rows": 0, "error": "no rows match those filters"}

    by_temp = [{"TEMP_C": int(k), "fail_rate_pct": _fail_rate(s)}
               for k, s in df.groupby("TEMP_C")]
    by_vdd = [{"VDD": float(k), "fail_rate_pct": _fail_rate(s)}
              for k, s in df.groupby("VDD")]
    return _wrap(df, f, {
        "test": test_txt or "all",
        "by_temperature": by_temp,
        "by_voltage": by_vdd,
    })


# ----------------------------------------------------------------------
# 5. fail rate by die position
# ----------------------------------------------------------------------

def get_yield_by_region(csv, test_txt=None, filters=None):
    """
    Failure rate by die position on the wafer.

    Returns radial bands and angular octants SEPARATELY, and does not
    return the band x octant grid. A defect confined to one sector is
    diluted in either marginal alone; resolving it requires crossing the
    two, which is composition and therefore must come from filters
    rather than from the tool.
    """
    df, f = _apply(_load(csv), filters)
    if test_txt:
        df = df[df["TEST_TXT"] == test_txt]
    if df.empty:
        return {"rows": 0, "error": "no rows match those filters"}

    by_band = [{"region": str(k), "fail_rate_pct": _fail_rate(s)}
               for k, s in df.groupby("REGION", observed=True)]
    by_oct = [{"octant": str(k), "fail_rate_pct": _fail_rate(s)}
              for k, s in df.groupby("OCTANT", observed=True)]
    return _wrap(df, f, {
        "test": test_txt or "all",
        "band_edges_normalised_radius": _BANDS,
        "by_radial_band": by_band,
        "by_octant": by_oct,
    })


# ----------------------------------------------------------------------
# 6. distribution of measured values
# ----------------------------------------------------------------------

def _stats(sub):
    """Summary statistics plus margin to the nearest active limit."""
    v = sub["RESULT"]
    lo = float(sub["LO_LIMIT"].iloc[0])
    hi = float(sub["HI_LIMIT"].iloc[0])
    mean, std = float(v.mean()), float(v.std())

    # Distance to each limit expressed in standard deviations. The
    # smaller of the two is the margin that actually matters.
    m_lo = (mean - lo) / std if std > 0 else None
    m_hi = (hi - mean) / std if std > 0 else None
    margin = min(m_lo, m_hi) if (m_lo is not None) else None

    return {
        "n": int(len(v)),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "p05": round(float(v.quantile(0.05)), 4),
        "p50": round(float(v.quantile(0.50)), 4),
        "p95": round(float(v.quantile(0.95)), 4),
        "lo_limit": lo,
        "hi_limit": hi,
        "margin_sigma": round(margin, 2) if margin is not None else None,
        "nearest_limit": ("LO" if m_lo < m_hi else "HI") if margin is not None else None,
    }


def get_distribution_stats(csv, test_txt, group_by=None, filters=None):
    """
    Distribution of measured values for one test, optionally per group.

    Failure counts say a distribution crossed a limit. This says by how
    much and in which direction, which is what separates a genuine
    parametric shift from a limit that was set too tightly, and lets a
    mean shift be seen before it has produced any yield loss.

    margin_sigma is the distance from the mean to the nearest active
    limit, in standard deviations. Roughly 3.0 is nominal here; smaller
    means the population has moved toward a limit.
    """
    df, f = _apply(_load(csv), filters)
    df = df[df["TEST_TXT"] == test_txt]
    if df.empty:
        return {"rows": 0, "error": f"no rows for test '{test_txt}' with those filters"}

    units = str(df["UNITS"].iloc[0])

    if group_by is None:
        return _wrap(df, f, {"test": test_txt, "units": units, "overall": _stats(df)})

    if group_by not in GROUP_FIELDS + ["TEMP_C", "VDD", "REGION", "OCTANT"]:
        return {"error": f"group_by must be one of {GROUP_FIELDS + ['TEMP_C','VDD','REGION','OCTANT']}"}

    groups = []
    for key, sub in df.groupby(group_by, observed=True):
        if len(sub) < 10:            # too few to characterise a distribution
            continue
        row = {group_by: key.item() if isinstance(key, np.generic) else str(key)}
        row.update(_stats(sub))
        groups.append(row)
    groups.sort(key=lambda r: r["margin_sigma"] if r["margin_sigma"] is not None else 99)

    return _wrap(df, f, {"test": test_txt, "units": units,
                         "group_by": group_by, "groups": groups})


# ----------------------------------------------------------------------

FUNCTIONS = {
    "get_overall_yield": get_overall_yield,
    "get_failures_by_test": get_failures_by_test,
    "get_yield_summary": get_yield_summary,
    "get_fail_rate_by_condition": get_fail_rate_by_condition,
    "get_yield_by_region": get_yield_by_region,
    "get_distribution_stats": get_distribution_stats,
}


if __name__ == "__main__":
    import json, sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/defect_4d.csv"

    print("--- overall, no filters ---")
    print(json.dumps(get_overall_yield(csv), indent=2))

    print("\n--- region, filtered to the cell ---")
    print(json.dumps(get_yield_by_region(
        csv, "core_fmax",
        {"LOT_ID": ["L47", "L48"], "TEMP_C": 85, "VDD": 0.75}), indent=2)[:900])