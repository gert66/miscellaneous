"""
Commercial Fit Scoring
======================
Converts enriched model-signal fields (Step 3 output) into a Final
Commercial Fit Score and assigns a commercial tier.

Formula
-------
    Final Commercial Fit Score = 0.75 × ICP Similarity Score
                                + 0.25 × Company Size Score

Both component scores are on a 1–10 scale.

IMPORTANT — PLACEHOLDER VALUES
===============================
All numerical constants below (LEAN_MODEL_COEFFICIENTS, INTERCEPT,
_SIZE_BANDS, TIER_THRESHOLDS) are PLACEHOLDER values derived from the
signal-weight analysis in postprocess_enrichment_single.py and the
feature-importance ordering in icp_model_pipeline.py.

Replace them with the values from Results(3).xlsx once that file is
available.  Each constant is documented with its source.  The scoring
logic itself (functions) does not need to change — only the constants.

Input fields (from enrich_clients_claude.py Step 3):
  MODEL_SIGNAL_SCORE_FIELDS  — ordinal int 0–3  (sig_*, ti_*, *_strength_*)
  MODEL_SIGNAL_BINARY_FIELDS — binary int 0/1   (has_*, is_public, has_funding)
  lusha_api_employee_range or lusha_employee_range — string

Output columns added to df_enriched:
  model_probability        float  [0, 1]
  icp_similarity_score     float  [1, 10]
  company_size_score       int    {2, 4, 6, 8, 10}
  commercial_fit_score     float  [~1, 10]
  commercial_tier          str    "Tier 1" | "Tier 2" | "Tier 3" | "Pass"
"""

from __future__ import annotations

import math
import re

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — replace with values from Results(3).xlsx
# ─────────────────────────────────────────────────────────────────────────────

#: Logistic-regression intercept.
#: PLACEHOLDER — replace with fitted value from Results(3).xlsx.
INTERCEPT: float = -2.5

#: Lean model coefficients mapped to model-signal field names.
#: Ordinal score fields (int 0–3) and binary fields (int 0 or 1).
#: Signs reflect expected direction; magnitudes reflect signal weights
#: from postprocess_enrichment_single.py.
#: PLACEHOLDER — replace with fitted LR coefficients from Results(3).xlsx.
LEAN_MODEL_COEFFICIENTS: dict[str, float] = {
    # ── Ordinal score fields (0–3) ─────────────────────────────────────────
    "sig_intl_footprint_score":               0.50,
    "sig_foreign_hq_score":                   0.40,
    "sig_explicit_lnd_score":                 0.70,
    "sig_multicultural_score":                0.45,
    "sig_employer_branding_score":            0.20,
    "sig_rapid_growth_score":                 0.35,
    "sig_merger_acq_score":                   0.25,
    "sig_lnd_onboarding_score":               0.55,
    "ti_language_english_score":              0.60,
    "ti_onboarding_score":                    0.30,
    "ti_leadership_score":                    0.25,
    "ti_broader_professional_score":          0.20,
    "ti_team_collab_score":                   0.15,
    "ti_intercultural_score":                 0.35,
    "ti_negotiation_sales_score":             0.25,
    "competitor_signal_strength_score":       0.45,
    "language_competitor_strength_score":     0.75,   # strongest predictor
    "online_learning_signal_strength_score":  0.30,
    "lnd_platform_signal_strength_score":     0.35,
    "model_signal_overall_confidence_score":  0.10,
    # ── Binary fields (0 or 1) ─────────────────────────────────────────────
    "has_competitor_signal":                  0.50,
    "has_language_competitor":                0.90,   # strongest binary predictor
    "has_online_learning_signal":             0.30,
    "has_lnd_platform_signal":                0.35,
    "is_public":                              0.10,
    "has_funding":                            0.20,
    "model_signal_needs_manual_review":      -0.30,   # negative: data uncertainty
}

#: Employee-count bands → Company Size Score (1–10 scale, five levels).
#: Each tuple is (exclusive upper bound of employee count, score).
#: Checked in order; first band whose upper bound exceeds the parsed
#: midpoint wins.
#: PLACEHOLDER — confirm band boundaries against Results(3).xlsx.
_SIZE_BANDS: list[tuple[float, int]] = [
    (50.0,       2),   # < 50    employees → tiny
    (200.0,      4),   # 50–199  employees → small
    (1_000.0,    6),   # 200–999 employees → medium
    (5_000.0,    8),   # 1k–5k   employees → large
    (math.inf,  10),   # > 5000  employees → enterprise
]

#: Commercial tier thresholds.  Tuple of (inclusive_lower_bound, tier_label).
#: Checked in order; the first entry whose bound ≤ fit_score wins.
#: PLACEHOLDER — replace thresholds from Results(3).xlsx.
TIER_THRESHOLDS: list[tuple[float, str]] = [
    (7.5, "Tier 1"),
    (6.0, "Tier 2"),
    (4.5, "Tier 3"),
    (0.0, "Pass"),
]

# Column names added by apply_scoring() — exported for UI use.
SCORING_COLS: list[str] = [
    "model_probability",
    "icp_similarity_score",
    "company_size_score",
    "commercial_fit_score",
    "commercial_tier",
]

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_employee_midpoint(raw) -> float | None:
    """Return a numeric midpoint from an employee-range string.

    Handles formats such as "51-200", "201–500", "1001 to 5000", "10001+".
    Returns None when the string is empty, non-numeric, or missing.
    """
    s = str(raw or "").strip().lower().replace(",", "").replace("–", "-")
    if s in ("", "nan", "none", "unknown", "n/a"):
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


# ─────────────────────────────────────────────────────────────────────────────
# Public scoring functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_model_probability(row: dict) -> float:
    """Apply the lean logistic regression to model-signal fields.

    Iterates over LEAN_MODEL_COEFFICIENTS; missing or non-numeric values
    are treated as 0.  Returns a probability in [0, 1].
    """
    logit = INTERCEPT
    for field, coef in LEAN_MODEL_COEFFICIENTS.items():
        try:
            v = float(row.get(field, 0) or 0)
        except (TypeError, ValueError):
            v = 0.0
        logit += coef * v
    # Clamp to prevent math.exp overflow at extreme logit values.
    logit = max(-500.0, min(500.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def icp_similarity_score(prob: float) -> float:
    """Convert model probability [0, 1] to ICP Similarity Score [1, 10].

    Linear rescaling: score = 1 + 9 × prob.
    """
    return round(1.0 + 9.0 * max(0.0, min(1.0, prob)), 2)


def company_size_score(row: dict) -> int:
    """Return Company Size Score on a 1–10 scale from the enriched employee range.

    Reads lusha_api_employee_range first, then lusha_employee_range.
    Returns 4 (small-band default) when the range is missing or unparseable.
    """
    raw = (
        row.get("lusha_api_employee_range")
        or row.get("lusha_employee_range")
        or ""
    )
    mid = _parse_employee_midpoint(raw)
    if mid is None:
        return 4
    for upper, score in _SIZE_BANDS:
        if mid < upper:
            return score
    return _SIZE_BANDS[-1][1]


def compute_commercial_fit_score(icp_sim: float, sz_sc: int) -> float:
    """Final Commercial Fit Score = 0.75 × ICP Similarity Score + 0.25 × Company Size Score."""
    return round(0.75 * icp_sim + 0.25 * sz_sc, 2)


def assign_tier(fit_score: float) -> str:
    """Assign commercial tier from Final Commercial Fit Score.

    Uses TIER_THRESHOLDS (inclusive lower bounds, checked in order).
    """
    for threshold, tier in TIER_THRESHOLDS:
        if fit_score >= threshold:
            return tier
    return "Pass"


def score_row(row: dict) -> dict:
    """Score a single enriched row.

    Returns a dict with the five SCORING_COLS keys.
    """
    prob    = compute_model_probability(row)
    icp_sim = icp_similarity_score(prob)
    sz_sc   = company_size_score(row)
    fit_sc  = compute_commercial_fit_score(icp_sim, sz_sc)
    tier    = assign_tier(fit_sc)
    return {
        "model_probability":    round(prob, 4),
        "icp_similarity_score": icp_sim,
        "company_size_score":   sz_sc,
        "commercial_fit_score": fit_sc,
        "commercial_tier":      tier,
    }


def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Apply commercial fit scoring to an enriched DataFrame.

    Appends the five SCORING_COLS columns.  Operates on a copy of each
    row's dict so the original DataFrame dtypes are not disturbed.
    Returns the same DataFrame object with the new columns added.
    """
    rows = df.to_dict("records")
    scored = [score_row(r) for r in rows]
    for col in SCORING_COLS:
        df[col] = [r[col] for r in scored]
    return df
