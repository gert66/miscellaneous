"""
Commercial Fit Scoring Engine
==============================
Standalone Python module that converts enriched model-signal fields into a
Final Commercial Fit Score and assigns a commercial tier.

Formula
-------
    lean_model_logit         = INTERCEPT + Σ (coeff_i × field_i)
    model_probability        = sigmoid(lean_model_logit)
    icp_similarity_score     = 1 + 9 × model_probability          (scale 1–10)
    company_size_score       = band lookup or passthrough          (scale 1–10)
    final_commercial_fit_score = 0.75 × icp_similarity_score
                               + 0.25 × company_size_score

Composite profile scores (0–10, normalised ordinal sums)
---------------------------------------------------------
Derived from signal groupings in postprocess_enrichment_single.py:
    global_complexity_score     — international footprint + multicultural signals
    people_development_score    — L&D focus + employer branding + leadership
    commercial_complexity_score — international presence + sales/language signals

Lean model input fields (7 ordinal signals, 0–3 each)
------------------------------------------------------
Copied from Results(3).xlsx — lean logistic regression feature set.
Replace INTERCEPT and LEAN_COEFFICIENTS below if the workbook is updated.

Usage
-----
    from commercial_fit_scoring import score_company, score_dataframe

    result = score_company(row_dict)
    df_out = score_dataframe(df_enriched)
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

# =============================================================================
# CONSTANTS
# Copied from Results(3).xlsx — lean logistic regression, 7-feature model.
# Replace these values when the workbook is updated.
# =============================================================================

#: Logistic-regression intercept.
#: Source: Results(3).xlsx — lean model fit.
INTERCEPT: float = -1.20

#: Lean model coefficients for the 7 selected input signals.
#: Keys are the enriched field names (ordinal int 0–3).
#: Source: Results(3).xlsx — lean model coefficients column.
LEAN_COEFFICIENTS: dict[str, float] = {
    "sig_foreign_hq_score":    0.48,
    "sig_rapid_growth_score":  0.42,
    "sig_explicit_lnd_score":  0.85,   # strongest predictor
    "sig_intl_footprint_score": 0.60,
    "sig_merger_acq_score":    0.30,
    "sig_lnd_onboarding_score": 0.65,
    "ti_onboarding_score":     0.38,
}

#: Employee-count bands → company_size_score on 1–10 scale.
#: Tuple: (exclusive upper bound of employee midpoint, score).
#: Source: Results(3).xlsx — company size scoring bands.
SIZE_BANDS: list[tuple[float, int]] = [
    (50.0,      2),    # <50       → micro/tiny
    (200.0,     4),    # 50–199    → small
    (1_000.0,   6),    # 200–999   → mid-market
    (5_000.0,   8),    # 1 000–4 999 → large
    (math.inf, 10),    # 5 000+    → enterprise
]

#: Tier thresholds — inclusive lower bounds, checked in order.
#: Source: Results(3).xlsx — commercial tier cut-offs.
TIER_THRESHOLDS: list[tuple[float, str]] = [
    (7.5, "Tier 1"),
    (6.0, "Tier 2"),
    (4.5, "Tier 3"),
    (0.0, "Pass"),
]

#: A contribution is flagged as a "top driver" when it exceeds this share of
#: the total positive logit mass.
TOP_DRIVER_THRESHOLD: float = 0.15

#: A field is "weak" when its value is 0 (or missing) but the coefficient
#: is at least this large — meaning scoring potential is being left unused.
WEAK_DRIVER_MIN_COEFF: float = 0.40

# =============================================================================
# COMPOSITE SCORE FIELD GROUPS
# Field groupings adapted from postprocess_enrichment_single.py signal analysis.
# Each group produces a normalised 0–10 composite score.
# =============================================================================

#: Captures how internationally complex the company's workforce and structure is.
#: Corresponds to global_complexity_score in postprocess_enrichment_single.py.
GLOBAL_COMPLEXITY_FIELDS: list[str] = [
    "sig_intl_footprint_score",   # international offices / global operations
    "sig_foreign_hq_score",       # non-domestic HQ or parent company
    "sig_multicultural_score",    # multilingual / diverse workforce signals
    "ti_intercultural_score",     # training interest in cross-cultural skills
]

#: Captures investment in people development, learning, and leadership.
#: Corresponds to people_development_score in postprocess_enrichment_single.py.
PEOPLE_DEVELOPMENT_FIELDS: list[str] = [
    "sig_explicit_lnd_score",     # explicit L&D programmes
    "sig_lnd_onboarding_score",   # structured onboarding / induction signals
    "sig_employer_branding_score",# employer brand / EVP investment
    "ti_leadership_score",        # training interest in leadership
    "ti_onboarding_score",        # training interest in onboarding
]

#: Captures commercial reach, cross-border sales complexity, and competitor signals.
#: Corresponds to commercial_complexity_score in postprocess_enrichment_single.py.
COMMERCIAL_COMPLEXITY_FIELDS: list[str] = [
    "sig_intl_footprint_score",            # international operations
    "ti_leadership_score",                 # sales / management training interest
    "ti_negotiation_sales_score",          # negotiation / sales training signals
    "language_competitor_strength_score",  # competitor uses language training
    "competitor_signal_strength_score",    # direct competitor signal
]

#: final_commercial_fit_score >= this value → high_value_flag = True (Tier 1 or 2).
HIGH_VALUE_MIN_SCORE: float = 6.0

#: final_commercial_fit_score < this value → weak_flag = True (Pass tier).
WEAK_MAX_SCORE: float = 4.5

#: Missing lean-model fields at or above this count → data_quality_flag = "medium".
DATA_QUALITY_MEDIUM_MISSING: int = 2

#: Missing lean-model fields at or above this count → data_quality_flag = "low".
DATA_QUALITY_LOW_MISSING: int = 5

# Output columns appended by score_dataframe()
SCORE_OUTPUT_COLS: list[str] = [
    "lean_model_logit",
    "model_probability",
    "icp_similarity_score",
    "company_size_score",
    "final_commercial_fit_score",
    "commercial_tier",
    "global_complexity_score",
    "people_development_score",
    "commercial_complexity_score",
    "high_value_flag",
    "weak_flag",
    "data_quality_flag",
    "top_score_drivers",
    "weak_score_drivers",
    "scoring_notes",
    "missing_scoring_fields",
]

# =============================================================================
# Internal helpers
# =============================================================================


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce value to float; return default on failure."""
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _is_missing(value: Any) -> bool:
    """Return True when value is absent, empty, or an explicit null string."""
    if value is None:
        return True
    s = str(value).strip().lower()
    return s in ("", "nan", "none", "n/a", "unknown", "null")


def _parse_employee_midpoint(raw: Any) -> float | None:
    """Return numeric midpoint from an employee-range string.

    Handles: "51-200", "201–500", "1001 to 5000", "10001+", plain integer.
    Returns None when the value is missing or cannot be parsed.
    """
    s = str(raw or "").strip().lower().replace(",", "").replace("–", "-")
    if s in ("", "nan", "none", "n/a", "unknown", "null"):
        return None
    # "201-500" or "201 to 500"
    m = re.search(r"(\d+)\s*[-to]+\s*(\d+)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    # "10001+"
    m = re.search(r"(\d+)\+", s)
    if m:
        return float(m.group(1)) * 1.5
    # plain integer
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return None


def _band_lookup(midpoint: float) -> int:
    """Map employee midpoint to size score via SIZE_BANDS."""
    for upper, score in SIZE_BANDS:
        if midpoint < upper:
            return score
    return SIZE_BANDS[-1][1]


def _composite_score(row: dict, fields: list[str], max_per_field: float = 3.0) -> float:
    """Normalised sum of ordinal fields, rescaled to 0–10.

    Missing or non-numeric values contribute 0.  Result is capped at 10.
    """
    total   = sum(_to_float(row.get(f, 0)) for f in fields)
    ceiling = max_per_field * len(fields)
    if ceiling <= 0:
        return 0.0
    return round(min(total / ceiling * 10.0, 10.0), 1)


# =============================================================================
# Public API
# =============================================================================


def score_company(
    row: dict | pd.Series,
    params: dict | None = None,
) -> dict:
    """Score a single enriched company row.

    Parameters
    ----------
    row:
        A dict or pandas.Series containing enriched model-signal fields.
        Missing fields are treated as 0 and logged in ``missing_scoring_fields``.
    params:
        Optional override dict.  Recognised keys:
          ``intercept``         — float, overrides INTERCEPT
          ``coefficients``      — dict[str, float], merges into LEAN_COEFFICIENTS
          ``size_bands``        — list[tuple[float, int]]
          ``tier_thresholds``   — list[tuple[float, str]]

    Returns
    -------
    dict with keys matching SCORE_OUTPUT_COLS:
      lean_model_logit, model_probability, icp_similarity_score,
      company_size_score, final_commercial_fit_score, commercial_tier,
      global_complexity_score, people_development_score,
      commercial_complexity_score, high_value_flag, weak_flag,
      data_quality_flag, top_score_drivers, weak_score_drivers,
      scoring_notes, missing_scoring_fields.
    """
    # ── Resolve parameters ────────────────────────────────────────────────────
    p = params or {}
    intercept  = float(p.get("intercept", INTERCEPT))
    coeffs     = {**LEAN_COEFFICIENTS, **p.get("coefficients", {})}
    size_bands = p.get("size_bands", SIZE_BANDS)
    tiers      = p.get("tier_thresholds", TIER_THRESHOLDS)

    if isinstance(row, pd.Series):
        row = row.to_dict()

    notes: list[str] = []
    missing_fields: list[str] = []

    # ── 1. Lean model logit ───────────────────────────────────────────────────
    logit = intercept
    field_contributions: dict[str, float] = {}

    for field, coef in coeffs.items():
        raw = row.get(field)
        if _is_missing(raw):
            missing_fields.append(field)
            v = 0.0
        else:
            v = _to_float(raw)
        contribution = coef * v
        field_contributions[field] = contribution
        logit += contribution

    logit = max(-500.0, min(500.0, logit))

    if missing_fields:
        notes.append(
            f"Score based on incomplete data — {len(missing_fields)} of "
            f"{len(coeffs)} signal field(s) missing (defaulted to 0): "
            f"{', '.join(missing_fields)}."
        )

    # ── 2. Model probability ──────────────────────────────────────────────────
    prob = 1.0 / (1.0 + math.exp(-logit))

    # ── 3. ICP Similarity Score (1–10) ────────────────────────────────────────
    icp_sim = round(1.0 + 9.0 * prob, 2)

    # ── 4. Company Size Score ─────────────────────────────────────────────────
    # Priority: company_size_score (1-10) > employee_size_score (1-5, scaled)
    # > derive from employee range string fields.
    size_score: int
    size_source: str

    if not _is_missing(row.get("company_size_score")):
        # Already on a 1–10 scale
        size_score = max(1, min(10, round(_to_float(row["company_size_score"]))))
        size_source = "company_size_score (passthrough)"

    elif not _is_missing(row.get("employee_size_score")):
        # Lusha 1–5 scale → convert to 1–10
        raw_ess = _to_float(row["employee_size_score"])
        size_score = max(1, min(10, round(raw_ess * 2)))
        size_source = "employee_size_score (scaled ×2)"

    else:
        # Derive from employee range string
        mid: float | None = None
        for field in (
            "lusha_api_employee_range",
            "lusha_employee_range",
            "employee_range",
            "company_size",
        ):
            raw_range = row.get(field)
            if not _is_missing(raw_range):
                mid = _parse_employee_midpoint(raw_range)
                if mid is not None:
                    size_source = f"derived from {field}"
                    break
        else:
            size_source = "default (no range data)"

        if mid is not None:
            size_score = _band_lookup(mid)
        else:
            size_score = 4   # conservative default (small-band)
            notes.append(
                "No employee count or range data found; company_size_score "
                "defaulted to 4 (small)."
            )

    # ── 5. Final Commercial Fit Score ─────────────────────────────────────────
    fit_score = round(0.75 * icp_sim + 0.25 * size_score, 2)

    # ── 6. Commercial Tier ────────────────────────────────────────────────────
    tier = "Pass"
    for threshold, label in tiers:
        if fit_score >= threshold:
            tier = label
            break

    # ── 7. Driver analysis ────────────────────────────────────────────────────
    total_positive = sum(c for c in field_contributions.values() if c > 0) or 1.0

    top_drivers: list[str] = [
        f"{field}={round(row.get(field, 0) or 0, 0):.0f} "
        f"(+{contribution:.2f})"
        for field, contribution in sorted(
            field_contributions.items(), key=lambda x: -x[1]
        )
        if contribution > 0
        and contribution / total_positive >= TOP_DRIVER_THRESHOLD
    ]

    # Weak drivers: high-coefficient fields the company scored 0 or is missing
    weak_drivers: list[str] = [
        f"{field} (coeff={coef:.2f}, current value={round(row.get(field, 0) or 0, 0):.0f})"
        for field, coef in sorted(coeffs.items(), key=lambda x: -x[1])
        if coef >= WEAK_DRIVER_MIN_COEFF
        and _to_float(row.get(field, 0)) == 0.0
    ]

    notes.append(f"Company size score: {size_score}/10 — {size_source}.")
    if tier == "Pass":
        notes.append("Score is below minimum tier threshold.")

    # ── 8. Composite profile scores ───────────────────────────────────────────
    global_complexity     = _composite_score(row, GLOBAL_COMPLEXITY_FIELDS)
    people_development    = _composite_score(row, PEOPLE_DEVELOPMENT_FIELDS)
    commercial_complexity = _composite_score(row, COMMERCIAL_COMPLEXITY_FIELDS)

    # ── 9. Value flags & data quality ─────────────────────────────────────────
    high_value_flag = fit_score >= HIGH_VALUE_MIN_SCORE
    weak_flag       = fit_score < WEAK_MAX_SCORE

    n_missing = len(missing_fields)
    manual_review = _to_float(row.get("model_signal_needs_manual_review", 0)) > 0
    if manual_review or n_missing >= DATA_QUALITY_LOW_MISSING:
        data_quality_flag = "low"
    elif n_missing >= DATA_QUALITY_MEDIUM_MISSING:
        data_quality_flag = "medium"
    else:
        data_quality_flag = "high"

    if data_quality_flag == "low" and not manual_review:
        notes.append(
            f"Data quality is low — {n_missing} of {len(coeffs)} signal fields missing."
        )
    elif manual_review:
        notes.append("Flagged for manual review; score reliability is reduced.")

    return {
        "lean_model_logit":           round(logit, 4),
        "model_probability":          round(prob, 4),
        "icp_similarity_score":       icp_sim,
        "company_size_score":         size_score,
        "final_commercial_fit_score": fit_score,
        "commercial_tier":            tier,
        "global_complexity_score":    global_complexity,
        "people_development_score":   people_development,
        "commercial_complexity_score": commercial_complexity,
        "high_value_flag":            high_value_flag,
        "weak_flag":                  weak_flag,
        "data_quality_flag":          data_quality_flag,
        "top_score_drivers":          "; ".join(top_drivers) if top_drivers else "none",
        "weak_score_drivers":         "; ".join(weak_drivers) if weak_drivers else "none",
        "scoring_notes":              " | ".join(notes),
        "missing_scoring_fields":     ", ".join(missing_fields) if missing_fields else "",
    }


def score_dataframe(
    df: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Apply score_company to every row and append score output columns.

    Parameters
    ----------
    df:
        DataFrame containing enriched model-signal fields.  Original columns
        are preserved; SCORE_OUTPUT_COLS are appended (overwriting any
        existing columns with those names).
    params:
        Passed through to score_company unchanged.

    Returns
    -------
    The same DataFrame object with SCORE_OUTPUT_COLS appended.
    """
    records = df.to_dict("records")
    scored  = [score_company(r, params) for r in records]
    for col in SCORE_OUTPUT_COLS:
        df[col] = [r[col] for r in scored]
    return df


# =============================================================================
# Unit-style tests — run with: python commercial_fit_scoring.py
# =============================================================================

if __name__ == "__main__":
    import json

    PASS  = "\033[92m✓\033[0m"
    FAIL  = "\033[91m✗\033[0m"
    _failures: list[str] = []

    def _check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  {PASS}  {label}")
        else:
            _failures.append(label)
            print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))

    def _section(title: str) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print("─" * 60)

    # ── Case 1: Strong company — all signals at maximum ───────────────────────
    _section("Case 1: Strong company (all signals = 3, large employee range)")

    strong = {
        "sig_foreign_hq_score":      3,
        "sig_rapid_growth_score":    3,
        "sig_explicit_lnd_score":    3,
        "sig_intl_footprint_score":  3,
        "sig_merger_acq_score":      3,
        "sig_lnd_onboarding_score":  3,
        "ti_onboarding_score":       3,
        "lusha_employee_range":      "5001-10000",
    }
    r1 = score_company(strong)
    print(json.dumps(r1, indent=2))

    _check("model_probability > 0.9",    r1["model_probability"] > 0.9,
           str(r1["model_probability"]))
    _check("icp_similarity_score > 9",   r1["icp_similarity_score"] > 9,
           str(r1["icp_similarity_score"]))
    _check("company_size_score == 10",   r1["company_size_score"] == 10,
           str(r1["company_size_score"]))
    _check("commercial_tier == Tier 1",  r1["commercial_tier"] == "Tier 1",
           r1["commercial_tier"])
    _check("top_score_drivers not empty", r1["top_score_drivers"] != "none")
    _check("missing_scoring_fields empty", r1["missing_scoring_fields"] == "")

    # ── Case 2: Weak company — all signals at minimum ─────────────────────────
    _section("Case 2: Weak company (all signals = 0, tiny employee range)")

    weak = {
        "sig_foreign_hq_score":      0,
        "sig_rapid_growth_score":    0,
        "sig_explicit_lnd_score":    0,
        "sig_intl_footprint_score":  0,
        "sig_merger_acq_score":      0,
        "sig_lnd_onboarding_score":  0,
        "ti_onboarding_score":       0,
        "lusha_employee_range":      "1-10",
    }
    r2 = score_company(weak)
    print(json.dumps(r2, indent=2))

    _check("model_probability < 0.5",    r2["model_probability"] < 0.5,
           str(r2["model_probability"]))
    _check("commercial_tier == Pass",    r2["commercial_tier"] == "Pass",
           r2["commercial_tier"])
    _check("company_size_score == 2",    r2["company_size_score"] == 2,
           str(r2["company_size_score"]))
    _check("weak_score_drivers not empty", r2["weak_score_drivers"] != "none")
    _check("missing_scoring_fields empty", r2["missing_scoring_fields"] == "")

    # ── Case 3: Company with missing values ───────────────────────────────────
    _section("Case 3: Company with missing/partial values")

    partial = {
        "sig_explicit_lnd_score":    2,    # only one field present
        "sig_intl_footprint_score":  None, # explicit None
        "lusha_employee_range":      "",   # empty string
        # all other signal fields absent
    }
    r3 = score_company(partial)
    print(json.dumps(r3, indent=2))

    _check("missing_scoring_fields not empty",
           r3["missing_scoring_fields"] != "")
    _check("scoring_notes mentions missing data",
           "missing" in r3["scoring_notes"].lower())
    _check("company_size_score == 4 (default)",
           r3["company_size_score"] == 4,
           str(r3["company_size_score"]))
    _check("model_probability is a valid float",
           0.0 <= r3["model_probability"] <= 1.0)
    _check("final_commercial_fit_score is numeric",
           isinstance(r3["final_commercial_fit_score"], float))

    # ── Case 4: Company with existing company_size_score ─────────────────────
    _section("Case 4: Company with pre-computed company_size_score")

    with_css = {
        "sig_foreign_hq_score":      2,
        "sig_explicit_lnd_score":    2,
        "sig_intl_footprint_score":  2,
        "sig_lnd_onboarding_score":  1,
        "company_size_score":        8,    # pre-computed, 1–10
    }
    r4 = score_company(with_css)
    print(json.dumps(r4, indent=2))

    _check("company_size_score passthrough == 8",
           r4["company_size_score"] == 8,
           str(r4["company_size_score"]))
    _check("size source note includes 'passthrough'",
           "passthrough" in r4["scoring_notes"])

    # ── Case 5: Company with employee_size_score (Lusha 1-5 scale) ───────────
    _section("Case 5: Company with employee_size_score (Lusha 1–5 scale)")

    with_ess = {
        "sig_explicit_lnd_score":    3,
        "sig_intl_footprint_score":  2,
        "sig_lnd_onboarding_score":  2,
        "ti_onboarding_score":       1,
        "employee_size_score":       4,    # Lusha 1-5 → should become 8
    }
    r5 = score_company(with_ess)
    print(json.dumps(r5, indent=2))

    _check("employee_size_score 4 scales to company_size_score 8",
           r5["company_size_score"] == 8,
           str(r5["company_size_score"]))

    # ── Case 6: score_dataframe ───────────────────────────────────────────────
    _section("Case 6: score_dataframe on a small DataFrame")

    test_df = pd.DataFrame([strong, weak, partial])
    out_df  = score_dataframe(test_df.copy())

    _check("score_dataframe returns DataFrame",
           isinstance(out_df, pd.DataFrame))
    _check("all SCORE_OUTPUT_COLS present",
           all(c in out_df.columns for c in SCORE_OUTPUT_COLS))
    _check("row count unchanged",
           len(out_df) == len(test_df))

    # ── Case 7: params override ───────────────────────────────────────────────
    _section("Case 7: params override (custom intercept)")

    custom = score_company(strong, params={"intercept": -99.0})
    _check("params override intercept drives prob to ~0",
           custom["model_probability"] < 0.01,
           str(custom["model_probability"]))
    _check("custom intercept → Pass tier",
           custom["commercial_tier"] == "Pass",
           custom["commercial_tier"])

    # ── Case 8: Composite scores — strong international company ──────────────
    _section("Case 8: Composite scores from a globally complex company")

    global_co = {
        "sig_intl_footprint_score":            3,
        "sig_foreign_hq_score":                3,
        "sig_multicultural_score":             3,
        "ti_intercultural_score":              3,
        "sig_explicit_lnd_score":              2,
        "sig_lnd_onboarding_score":            2,
        "sig_employer_branding_score":         1,
        "ti_leadership_score":                 2,
        "ti_onboarding_score":                 1,
        "ti_negotiation_sales_score":          2,
        "language_competitor_strength_score":  3,
        "competitor_signal_strength_score":    2,
        "lusha_employee_range":                "1001-5000",
    }
    r8 = score_company(global_co)
    print(json.dumps(r8, indent=2))

    _check("global_complexity_score == 10.0 (all 4 fields maxed)",
           r8["global_complexity_score"] == 10.0,
           str(r8["global_complexity_score"]))
    _check("people_development_score > 5",
           r8["people_development_score"] > 5,
           str(r8["people_development_score"]))
    _check("commercial_complexity_score > 5",
           r8["commercial_complexity_score"] > 5,
           str(r8["commercial_complexity_score"]))
    _check("high_value_flag is True",
           r8["high_value_flag"] is True,
           str(r8["high_value_flag"]))
    _check("weak_flag is False",
           r8["weak_flag"] is False,
           str(r8["weak_flag"]))

    # ── Case 9: Data quality flags ────────────────────────────────────────────
    _section("Case 9: Data quality flags")

    # High quality — all 7 lean fields present
    r9a = score_company(strong)
    _check("full signals → data_quality_flag == 'high'",
           r9a["data_quality_flag"] == "high",
           r9a["data_quality_flag"])

    # Medium quality — 3 lean fields missing (4 of 7 present)
    r9b = score_company({
        "sig_explicit_lnd_score":   2,
        "sig_lnd_onboarding_score": 1,
        "sig_intl_footprint_score": 2,
        "sig_foreign_hq_score":     1,
        "lusha_employee_range":     "201-500",
    })
    _check("3 missing lean fields → data_quality_flag == 'medium'",
           r9b["data_quality_flag"] == "medium",
           r9b["data_quality_flag"])

    # Low quality — all lean fields absent
    r9c = score_company({"lusha_employee_range": "201-500"})
    _check("all lean fields missing → data_quality_flag == 'low'",
           r9c["data_quality_flag"] == "low",
           r9c["data_quality_flag"])

    # Manual review flag → always low quality regardless of missing count
    r9d = score_company({**strong, "model_signal_needs_manual_review": 1})
    _check("manual_review=1 → data_quality_flag == 'low'",
           r9d["data_quality_flag"] == "low",
           r9d["data_quality_flag"])
    _check("manual_review=1 → scoring_notes mentions manual review",
           "manual review" in r9d["scoring_notes"].lower())

    # ── Case 10: weak_flag and high_value_flag boundaries ────────────────────
    _section("Case 10: high_value_flag / weak_flag boundary values")

    # Exact Pass boundary: all signals 0, default size 4 → fit ≈ 0.75*icp + 1.0
    r10a = score_company(weak)
    _check("weak company → weak_flag is True",
           r10a["weak_flag"] is True,
           str(r10a["weak_flag"]))
    _check("weak company → high_value_flag is False",
           r10a["high_value_flag"] is False,
           str(r10a["high_value_flag"]))

    # Strong company → Tier 1
    r10b = score_company(strong)
    _check("strong company → high_value_flag is True",
           r10b["high_value_flag"] is True,
           str(r10b["high_value_flag"]))
    _check("strong company → weak_flag is False",
           r10b["weak_flag"] is False,
           str(r10b["weak_flag"]))

    # ── Case 11: score_dataframe includes all SCORE_OUTPUT_COLS ──────────────
    _section("Case 11: score_dataframe — all new columns present")

    df11 = pd.DataFrame([strong, weak, global_co])
    out11 = score_dataframe(df11.copy())
    _check("all SCORE_OUTPUT_COLS in dataframe",
           all(c in out11.columns for c in SCORE_OUTPUT_COLS),
           str([c for c in SCORE_OUTPUT_COLS if c not in out11.columns]))
    _check("global_complexity_score column is numeric",
           pd.to_numeric(out11["global_complexity_score"], errors="coerce").notna().all())
    _check("high_value_flag column is bool",
           out11["high_value_flag"].dtype == bool)
    _check("data_quality_flag values are in expected set",
           out11["data_quality_flag"].isin({"high", "medium", "low"}).all())

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    if _failures:
        print(f"  FAILURES ({len(_failures)}):")
        for f in _failures:
            print(f"    • {f}")
    else:
        print("  All checks passed.")
    print("═" * 60)
