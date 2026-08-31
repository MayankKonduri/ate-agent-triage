"""
tools.py — JSON schemas, ablation configurations, and dispatch.

Three responsibilities:

  SCHEMAS   one JSON schema per query function in analyze.py
  CONFIGS   which functions, and whether `filters` is exposed
  dispatch  route a tool call from the model to the Python function

The ablation lives in CONFIGS. `reduced` and `full` expose the SAME six
functions; they differ only in whether the `filters` property appears in
the schema. That isolates one capability -- composing conditions into a
single query -- from the confound of simply having more tools.

  minimal   2 functions, no filters   -> no grouping axis at all
  reduced   6 functions, no filters   -> every axis, marginals only
  full      6 functions, with filters -> every axis, can intersect
"""

import copy
import analyze

# ----------------------------------------------------------------------
# the filters property, added to a schema only in the `full` config
# ----------------------------------------------------------------------

_FILTERS = {
    "type": "object",
    "description": (
        "Optional. Restrict which rows are considered BEFORE the "
        "aggregation is computed. Each key is a column and each value is "
        "a single value or a list of values. Use this to carry a finding "
        "from one query into the next, e.g. after finding a suspect lot, "
        "pass {\"LOT_ID\": [\"L47\"]} to look only within it."
    ),
    "properties": {
        "LOT_ID":    {"description": "e.g. \"L47\" or [\"L47\", \"L48\"]"},
        "WAFER_ID":  {"description": "e.g. \"W03\""},
        "SITE_NUM":  {"description": "test site, 1-8"},
        "TEST_TXT":  {"description": "test name"},
        "TEMP_C":    {"description": "insertion temperature: 0, 25 or 85"},
        "VDD":       {"description": "supply voltage: 0.75, 0.80 or 0.85"},
        "INSERTION": {"description": "sort_cold, sort_room or sort_hot"},
        "REGION":    {"description": "radial band: centre, mid, outer or edge"},
    },
}


def _schema(name, description, properties=None, required=None):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
    }


# ----------------------------------------------------------------------
# schemas — one per function in analyze.py
# ----------------------------------------------------------------------

SCHEMAS = {

    "get_overall_yield": _schema(
        "get_overall_yield",
        "Part-level yield and measurement failure rate. A die counts as "
        "failed if any of its measurements is outside limits. Use to size "
        "an excursion, or to check how concentrated a subset is.",
    ),

    "get_failures_by_test": _schema(
        "get_failures_by_test",
        "Failure count and rate for each of the ten tests, ranked worst "
        "first (a Pareto). Says WHICH tests are failing, not where or "
        "under what conditions.",
    ),

    "get_yield_summary": _schema(
        "get_yield_summary",
        "Part-level yield grouped by one categorical field. Use LOT_ID to "
        "check for a batch effect, WAFER_ID for a wafer effect, SITE_NUM "
        "to check whether one tester site is responsible.",
        {"group_by": {"type": "string",
                      "enum": ["LOT_ID", "WAFER_ID", "SITE_NUM"],
                      "description": "field to group by"}},
        ["group_by"],
    ),

    "get_fail_rate_by_condition": _schema(
        "get_fail_rate_by_condition",
        "Failure rate split by test temperature and supply voltage. "
        "Returns each marginal plus the full TEMP_C x VDD grid, which "
        "distinguishes an interaction between the two from two "
        "independent effects.",
        {"test_txt": {"type": "string",
                      "description": "test name; omit to pool all tests"}},
    ),

    "get_yield_by_region": _schema(
        "get_yield_by_region",
        "Failure rate by die position on the wafer. Returns radial bands "
        "(centre/mid/outer/edge), 45-degree octants (N, NE, E, ...), and "
        "the band x octant grid. A defect confined to one sector is "
        "diluted in either marginal alone and resolves only in the grid.",
        {"test_txt": {"type": "string",
                      "description": "test name; omit to pool all tests"}},
    ),

    "get_distribution_stats": _schema(
        "get_distribution_stats",
        "Distribution of measured values for one test: mean, standard "
        "deviation, percentiles, the limits, and margin_sigma (distance "
        "from the mean to the nearest limit, in standard deviations; "
        "about 3.0 is nominal). Shows how far and in which direction a "
        "population has moved, including shifts too small to have caused "
        "failures yet.",
        {"test_txt": {"type": "string", "description": "test name"},
         "group_by": {"type": "string",
                      "enum": ["LOT_ID", "WAFER_ID", "SITE_NUM",
                               "TEMP_C", "VDD", "REGION"],
                      "description": "optional field to group by"}},
        ["test_txt"],
    ),
}


# ----------------------------------------------------------------------
# ablation configurations
#
# A 2 x 2 design over two independent capabilities:
#
#                     |  no filters   |  filters
#   ------------------+---------------+--------------
#   2 functions       |  minimal      |  minimal_f
#   6 functions       |  reduced      |  full
#
# COVERAGE  (rows) is how many analysis axes the tools expose at all.
# COMPOSITION (cols) is whether findings can be combined into a single
# query. They are separable: minimal_f can reach any dimension, but only
# by probing one value at a time; reduced can group along every axis,
# but every result is a marginal averaged over the others.
# ----------------------------------------------------------------------

_ALL = list(SCHEMAS.keys())
_TWO = ["get_overall_yield", "get_failures_by_test"]

CONFIGS = {
    "minimal":   {"functions": _TWO,  "filters": False},
    "minimal_f": {"functions": _TWO,  "filters": True},
    "reduced":   {"functions": _ALL,  "filters": False},
    "full":      {"functions": _ALL,  "filters": True},
}


def build_tools(config):
    """
    Return the list of tool schemas exposed by a configuration.

    When the config allows filtering, the `filters` property is added to
    every schema. When it does not, the property is absent, so the model
    has no way to express a restricted query.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config '{config}'; expected {list(CONFIGS)}")

    cfg = CONFIGS[config]
    tools = []
    for name in cfg["functions"]:
        s = copy.deepcopy(SCHEMAS[name])
        if cfg["filters"]:
            s["input_schema"]["properties"]["filters"] = copy.deepcopy(_FILTERS)
        tools.append(s)
    return tools


# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------

def dispatch(name, args, csv, config):
    """
    Call the analyze.py function behind a tool name.

    The dataset path is injected here rather than passed by the model, so
    a run cannot be pointed at the wrong file. Filters are dropped if the
    configuration does not permit them, so a model that hallucinates the
    argument cannot get around the ablation.

    Never raises: any failure is returned as {"error": ...} so that one
    bad call does not end the run.
    """
    fn = analyze.FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}

    args = dict(args or {})
    if not CONFIGS[config]["filters"]:
        args.pop("filters", None)

    try:
        return fn(csv, **args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}


if __name__ == "__main__":
    import json
    for name, cfg in CONFIGS.items():
        tools = build_tools(name)
        has_f = [t["name"] for t in tools
                 if "filters" in t["input_schema"]["properties"]]
        print(f"{name:9s} {len(tools)} tools, filters on {len(has_f)}")
    print()
    print("dispatch, full config, filtered:")
    print(json.dumps(dispatch("get_overall_yield",
                              {"filters": {"LOT_ID": ["L47", "L48"],
                                           "TEMP_C": 85, "VDD": 0.75}},
                              "data/defect_4d.csv", "full")))
    print()
    print("same call, reduced config (filters silently dropped):")
    print(json.dumps(dispatch("get_overall_yield",
                              {"filters": {"LOT_ID": ["L47", "L48"],
                                           "TEMP_C": 85, "VDD": 0.75}},
                              "data/defect_4d.csv", "reduced")))