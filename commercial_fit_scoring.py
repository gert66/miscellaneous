"""
Commercial Fit Scoring Engine — Results(8).xlsx-compatible formula
==================================================================
Implements the logistic regression + sigmoid-stretch + size-blend formula
validated against Results(8).xlsx.

Formula summary
---------------
1.  Normalise each input signal:  norm = clamp(v, 0, 3) / 3
2.  LR score:   lr_z_score  = INTERCEPT + Σ(coeff_i × norm_i)
3.  Prob:        lean_model_prob = 1 / (1 + exp(−lr_z_score))
4.  Sigmoid stretch:
      sigmoid_raw_s       = 1 / (1 + exp(−k × (prob − 0.5)))
      icp_similarity_score = clamp(1 + 9×(s − s_min)/(s_max − s_min), 1, 10)
5.  Size score: exact band lookup (9 bands, 1–10 float scale)
6.  Blend:
      final_commercial_fit_score =
          clamp(0.75 × icp_similarity_score + 0.25 × company_size_score, 1, 10)

Backward-compatible aliases (calculated from canonical fields, not independently)
-----------------------------------------------------------------------------------
  lean_model_logit  → alias of lr_z_score
  model_probability → alias of lean_model_prob

Reference validation (Capgemini, all 7 signals supplied)
---------------------------------------------------------
  lr_z_score          ≈ 0.9870
  lean_model_prob     ≈ 0.7285
  icp_similarity_score ≈ 9.39
  company_size_score  = 10.0
  final_commercial_fit_score ≈ 9.54
  commercial_tier     = 🥇 Hot
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

# =============================================================================
# CONSTANTS — Results(8).xlsx-compatible
# =============================================================================

#: LR intercept.
INTERCEPT: float = -0.35

#: Coefficients applied to normalised signals (clamp(v, 0, 3) / 3).
#: IMPORTANT: uses sig_employer_branding_score — NOT sig_merger_acq_score.
LEAN_COEFFICIENTS: dict[str, float] = {
    "sig_foreign_hq_score":        0.7465,
    "sig_explicit_lnd_score":      0.2185,
    "sig_intl_footprint_score":    0.1795,
    "sig_employer_branding_score": 0.1602,   # NOT sig_merger_acq_score
    "sig_lnd_onboarding_score":    0.1250,
    "ti_onboarding_score":         0.1488,
    "sig_rapid_growth_score":     -0.2905,   # negative coefficient
}

#: Exact employee-range string → company_size_score (1–10 float).
#: Keys are the normalised canonical form (lower - upper).
SIZE_BAND_LOOKUP: dict[str, float] = {
    "100001 - 10000000": 10.0,
    "10001 - 100000":     8.88,
    "5001 - 10000":       7.75,
    "1001 - 5000":        6.63,
    "501 - 1000":          5.5,
    "201 - 500":           4.38,
    "51 - 200":            3.25,
    "11 - 50":             2.13,
    "1 - 10":              1.0,
}
#: Numeric midpoint thresholds for fallback band lookup (used when exact
#: string match fails).  Checked ascending; first band whose upper bound
#: exceeds the midpoint wins.
_SIZE_MIDPOINT_BANDS: list[tuple[float, float]] = [
    (10.5,      1.0),
    (50.5,      2.13),
    (200.5,     3.25),
    (500.5,     4.38),
    (1_000.5,   5.5),
    (5_000.5,   6.63),
    (10_000.5,  7.75),
    (100_000.5, 8.88),
    (math.inf,  10.0),
]
SIZE_SCORE_MISSING: float = 5.5   # default when range is unknown

#: Sigmoid stretch parameters.
SIGMOID_K:     float = 5.0
SIGMOID_S_MIN: float = 0.328827
SIGMOID_S_MAX: float = 0.789236

#: Blend weights.
MODEL_WEIGHT: float = 0.75
SIZE_WEIGHT:  float = 0.25

#: Tier thresholds — inclusive lower bounds, checked in descending order.
TIER_THRESHOLDS: list[tuple[float, str]] = [
    (8.66, "🥇 Hot"),
    (7.19, "🥈 Warm"),
    (4.23, "🥉 Cool"),
    (0.0,  "❄️ Pass"),
]

# Composite profile score groupings (display-only — not part of LR)
GLOBAL_COMPLEXITY_FIELDS: list[str] = [
    "sig_intl_footprint_score",
    "sig_foreign_hq_score",
    "sig_multicultural_score",
    "ti_intercultural_score",
]
PEOPLE_DEVELOPMENT_FIELDS: list[str] = [
    "sig_explicit_lnd_score",
    "sig_lnd_onboarding_score",
    "sig_employer_branding_score",
    "ti_leadership_score",
    "ti_onboarding_score",
]
COMMERCIAL_COMPLEXITY_FIELDS: list[str] = [
    "sig_intl_footprint_score",
    "ti_leadership_score",
    "ti_negotiation_sales_score",
    "language_competitor_strength_score",
    "competitor_signal_strength_score",
]

HIGH_VALUE_MIN_SCORE: float = 7.19   # Hot or Warm
WEAK_MAX_SCORE:       float = 4.23   # below Cool
DATA_QUALITY_MEDIUM_MISSING: int = 2
DATA_QUALITY_LOW_MISSING:    int = 5
TOP_DRIVER_THRESHOLD:   float = 0.15
WEAK_DRIVER_MIN_COEFF:  float = 0.10   # positive coefficients only

#: All columns appended by score_dataframe().
SCORE_OUTPUT_COLS: list[str] = [
    # ── Canonical result fields ──────────────────────────────────────────────
    "lr_z_score",
    "lean_model_prob",
    "icp_similarity_score",
    "company_size_score",
    "company_size_missing",
    "final_commercial_fit_score",
    "commercial_tier",
    # ── Backward-compat aliases ──────────────────────────────────────────────
    "lean_model_logit",       # = lr_z_score
    "model_probability",      # = lean_model_prob
    # ── Audit: LR raw inputs ─────────────────────────────────────────────────
    "score_input_foreign_hq",
    "score_input_explicit_lnd",
    "score_input_intl_footprint",
    "score_input_employer_branding",
    "score_input_lnd_onboarding",
    "score_input_ti_onboarding",
    "score_input_rapid_growth",
    "score_input_employee_range",
    # ── Audit: normalised inputs ─────────────────────────────────────────────
    "norm_foreign_hq",
    "norm_explicit_lnd",
    "norm_intl_footprint",
    "norm_employer_branding",
    "norm_lnd_onboarding",
    "norm_ti_onboarding",
    "norm_rapid_growth",
    # ── Audit: LR components ─────────────────────────────────────────────────
    "lr_intercept_component",
    "lr_foreign_hq_component",
    "lr_explicit_lnd_component",
    "lr_intl_footprint_component",
    "lr_employer_branding_component",
    "lr_lnd_onboarding_component",
    "lr_ti_onboarding_component",
    "lr_rapid_growth_component",
    # ── Audit: size ──────────────────────────────────────────────────────────
    "employee_range_normalized",
    # ── Audit: sigmoid ───────────────────────────────────────────────────────
    "sigmoid_k",
    "sigmoid_s_min",
    "sigmoid_s_max",
    "sigmoid_input_value",
    "sigmoid_raw_s",
    # ── Audit: blend ─────────────────────────────────────────────────────────
    "model_weight",
    "size_weight",
    "weighted_model_component",
    "weighted_size_component",
    # ── Composite profile scores (display-only) ───────────────────────────────
    "global_complexity_score",
    "people_development_score",
    "commercial_complexity_score",
    # ── Quality flags ────────────────────────────────────────────────────────
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
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in ("", "nan", "none", "n/a", "unknown", "null")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_range_str(raw: str) -> str:
    """Canonicalise 'lower-upper' range string to 'lo - hi' form."""
    s = raw.strip().lower().replace(",", "").replace("–", "-").replace(" to ", "-")
    s = re.sub(r"\s*-\s*", " - ", s)
    return s


def _parse_range_midpoint(raw: Any) -> float | None:
    """Return numeric midpoint of an employee-range string, or None."""
    s = str(raw or "").strip().lower().replace(",", "").replace("–", "-")
    if s in ("", "nan", "none", "n/a", "unknown", "null"):
        return None
    m = re.search(r"(\d+)\s*[-to]+\s*(\d+)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.search(r"(\d+)\+", s)
    if m:
        return float(m.group(1)) * 1.5
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return None


def _band_lookup_midpoint(midpoint: float) -> float:
    for upper, score in _SIZE_MIDPOINT_BANDS:
        if midpoint < upper:
            return score
    return 10.0


def _resolve_size_score(row: dict) -> tuple[float, bool, str]:
    """Return (size_score, size_missing, range_key).

    Tries lusha_api_employee_range then lusha_employee_range.
    Falls back to SIZE_SCORE_MISSING when nothing is parseable.
    """
    for field in ("lusha_api_employee_range", "lusha_employee_range",
                  "employee_range", "company_size"):
        raw = row.get(field)
        if _is_missing(raw):
            continue
        # Try exact normalised-key lookup first
        norm_key = _normalize_range_str(str(raw))
        if norm_key in SIZE_BAND_LOOKUP:
            return SIZE_BAND_LOOKUP[norm_key], False, norm_key
        # Try direct key match (handles already-canonical strings)
        direct = str(raw).strip()
        if direct in SIZE_BAND_LOOKUP:
            return SIZE_BAND_LOOKUP[direct], False, direct
        # Fall back to midpoint-based band
        mid = _parse_range_midpoint(raw)
        if mid is not None:
            return _band_lookup_midpoint(mid), False, norm_key
    return SIZE_SCORE_MISSING, True, ""


def _composite_score(row: dict, fields: list[str], max_per_field: float = 3.0) -> float:
    total   = sum(_to_float(row.get(f, 0)) for f in fields)
    ceiling = max_per_field * len(fields)
    if ceiling <= 0:
        return 0.0
    return round(min(total / ceiling * 10.0, 10.0), 1)


# =============================================================================
# Public API
# =============================================================================

_FIELD_TO_AUDIT_INPUT: dict[str, str] = {
    "sig_foreign_hq_score":        "score_input_foreign_hq",
    "sig_explicit_lnd_score":      "score_input_explicit_lnd",
    "sig_intl_footprint_score":    "score_input_intl_footprint",
    "sig_employer_branding_score": "score_input_employer_branding",
    "sig_lnd_onboarding_score":    "score_input_lnd_onboarding",
    "ti_onboarding_score":         "score_input_ti_onboarding",
    "sig_rapid_growth_score":      "score_input_rapid_growth",
}
_FIELD_TO_NORM: dict[str, str] = {
    "sig_foreign_hq_score":        "norm_foreign_hq",
    "sig_explicit_lnd_score":      "norm_explicit_lnd",
    "sig_intl_footprint_score":    "norm_intl_footprint",
    "sig_employer_branding_score": "norm_employer_branding",
    "sig_lnd_onboarding_score":    "norm_lnd_onboarding",
    "ti_onboarding_score":         "norm_ti_onboarding",
    "sig_rapid_growth_score":      "norm_rapid_growth",
}
_FIELD_TO_COMPONENT: dict[str, str] = {
    "sig_foreign_hq_score":        "lr_foreign_hq_component",
    "sig_explicit_lnd_score":      "lr_explicit_lnd_component",
    "sig_intl_footprint_score":    "lr_intl_footprint_component",
    "sig_employer_branding_score": "lr_employer_branding_component",
    "sig_lnd_onboarding_score":    "lr_lnd_onboarding_component",
    "ti_onboarding_score":         "lr_ti_onboarding_component",
    "sig_rapid_growth_score":      "lr_rapid_growth_component",
}


def score_company(
    row: "dict | pd.Series",
    params: dict | None = None,
) -> dict:
    """Score a single enriched company row using the Results(8)-compatible formula.

    Returns a dict whose keys are a superset of SCORE_OUTPUT_COLS, including
    all audit fields so callers can write them to a model_features sheet.
    """
    p = params or {}
    intercept = float(p.get("intercept", INTERCEPT))
    coeffs    = {**LEAN_COEFFICIENTS, **p.get("coefficients", {})}
    tiers     = p.get("tier_thresholds", TIER_THRESHOLDS)

    if isinstance(row, pd.Series):
        row = row.to_dict()

    notes: list[str]   = []
    missing: list[str] = []
    out: dict = {}

    # ── 1. Read raw signal values, normalise, compute LR components ──────────
    lr_z  = intercept
    out["lr_intercept_component"] = intercept

    for field, coeff in coeffs.items():
        raw = row.get(field)
        if _is_missing(raw):
            missing.append(field)
            raw_val = 0.0
        else:
            raw_val = _clamp(_to_float(raw), 0.0, 3.0)

        norm = raw_val / 3.0
        comp = coeff * norm
        lr_z += comp

        out[_FIELD_TO_AUDIT_INPUT[field]] = raw_val
        out[_FIELD_TO_NORM[field]]        = round(norm, 6)
        out[_FIELD_TO_COMPONENT[field]]   = round(comp, 6)

    if missing:
        notes.append(
            f"Score based on incomplete data — {len(missing)} of {len(coeffs)} "
            f"signal field(s) missing (defaulted to 0): {', '.join(missing)}."
        )

    lr_z = _clamp(lr_z, -500.0, 500.0)

    # ── 2. Probability ────────────────────────────────────────────────────────
    lean_model_prob = 1.0 / (1.0 + math.exp(-lr_z))

    # ── 3. Sigmoid stretch → ICP Similarity Score ────────────────────────────
    sigmoid_raw_s = 1.0 / (1.0 + math.exp(-SIGMOID_K * (lean_model_prob - 0.5)))
    denom = SIGMOID_S_MAX - SIGMOID_S_MIN
    icp_sim = _clamp(1.0 + 9.0 * (sigmoid_raw_s - SIGMOID_S_MIN) / denom, 1.0, 10.0)

    # ── 4. Size score ─────────────────────────────────────────────────────────
    size_score, size_missing, range_key = _resolve_size_score(row)
    out["score_input_employee_range"] = str(
        next((row.get(f) for f in ("lusha_api_employee_range", "lusha_employee_range",
                                   "employee_range", "company_size")
              if not _is_missing(row.get(f))), "")
    )
    out["employee_range_normalized"] = range_key
    if size_missing:
        notes.append(
            "No employee range data found; company_size_score defaulted to "
            f"{SIZE_SCORE_MISSING}."
        )

    # ── 5. Blend ──────────────────────────────────────────────────────────────
    w_model = MODEL_WEIGHT * icp_sim
    w_size  = SIZE_WEIGHT  * size_score
    final   = _clamp(w_model + w_size, 1.0, 10.0)

    # ── 6. Tier ───────────────────────────────────────────────────────────────
    tier = TIER_THRESHOLDS[-1][1]
    for threshold, label in tiers:
        if final >= threshold:
            tier = label
            break

    # ── 7. Composite profile scores ───────────────────────────────────────────
    global_complexity     = _composite_score(row, GLOBAL_COMPLEXITY_FIELDS)
    people_development    = _composite_score(row, PEOPLE_DEVELOPMENT_FIELDS)
    commercial_complexity = _composite_score(row, COMMERCIAL_COMPLEXITY_FIELDS)

    # ── 8. Driver analysis ────────────────────────────────────────────────────
    pos_contributions = {
        f: out[_FIELD_TO_COMPONENT[f]]
        for f in coeffs
        if out[_FIELD_TO_COMPONENT[f]] > 0
    }
    total_pos = sum(pos_contributions.values()) or 1.0

    top_drivers = [
        f"{f}={out[_FIELD_TO_AUDIT_INPUT[f]]:.0f} (+{c:.2f})"
        for f, c in sorted(pos_contributions.items(), key=lambda x: -x[1])
        if c / total_pos >= TOP_DRIVER_THRESHOLD
    ]
    weak_drivers = [
        f"{f} (coeff={coeffs[f]:.4f}, current={out[_FIELD_TO_AUDIT_INPUT[f]]:.0f})"
        for f in sorted(coeffs, key=lambda x: -coeffs[x])
        if coeffs[f] >= WEAK_DRIVER_MIN_COEFF
        and out[_FIELD_TO_AUDIT_INPUT[f]] == 0.0
    ]

    # ── 9. Data quality ───────────────────────────────────────────────────────
    n_missing  = len(missing)
    manual_rev = _to_float(row.get("model_signal_needs_manual_review", 0)) > 0
    if manual_rev or n_missing >= DATA_QUALITY_LOW_MISSING:
        dqf = "low"
    elif n_missing >= DATA_QUALITY_MEDIUM_MISSING:
        dqf = "medium"
    else:
        dqf = "high"
    if dqf == "low" and not manual_rev:
        notes.append(f"Data quality is low — {n_missing} of {len(coeffs)} signal fields missing.")
    elif manual_rev:
        notes.append("Flagged for manual review; score reliability is reduced.")

    notes.append(f"Company size score: {size_score}/10.")

    # ── Assemble result ───────────────────────────────────────────────────────
    out.update({
        # Canonical fields
        "lr_z_score":                  round(lr_z, 6),
        "lean_model_prob":             round(lean_model_prob, 7),
        "icp_similarity_score":        round(icp_sim, 2),
        "company_size_score":          round(size_score, 2),
        "company_size_missing":        size_missing,
        "final_commercial_fit_score":  round(final, 2),
        "commercial_tier":             tier,
        # Backward-compat aliases (derived from canonical — not a separate calculation)
        "lean_model_logit":            round(lr_z, 6),
        "model_probability":           round(lean_model_prob, 7),
        # Sigmoid audit
        "sigmoid_k":           SIGMOID_K,
        "sigmoid_s_min":       SIGMOID_S_MIN,
        "sigmoid_s_max":       SIGMOID_S_MAX,
        "sigmoid_input_value": round(lean_model_prob, 7),
        "sigmoid_raw_s":       round(sigmoid_raw_s, 7),
        # Blend audit
        "model_weight":              MODEL_WEIGHT,
        "size_weight":               SIZE_WEIGHT,
        "weighted_model_component":  round(w_model, 4),
        "weighted_size_component":   round(w_size, 4),
        # Composite
        "global_complexity_score":     global_complexity,
        "people_development_score":    people_development,
        "commercial_complexity_score": commercial_complexity,
        # Flags
        "high_value_flag":        final >= HIGH_VALUE_MIN_SCORE,
        "weak_flag":              final < WEAK_MAX_SCORE,
        "data_quality_flag":      dqf,
        "top_score_drivers":      "; ".join(top_drivers) if top_drivers else "none",
        "weak_score_drivers":     "; ".join(weak_drivers) if weak_drivers else "none",
        "scoring_notes":          " | ".join(notes),
        "missing_scoring_fields": ", ".join(missing) if missing else "",
    })
    return out


def score_dataframe(
    df: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Apply score_company to every row and append SCORE_OUTPUT_COLS."""
    records = df.to_dict("records")
    scored  = [score_company(r, params) for r in records]
    for col in SCORE_OUTPUT_COLS:
        df[col] = [r.get(col) for r in scored]
    return df


# =============================================================================
# Smoke tests — run with:  python commercial_fit_scoring.py
# =============================================================================

if __name__ == "__main__":
    import json

    PASS  = "\033[92m✓\033[0m"
    FAIL  = "\033[91m✗\033[0m"
    _failures: list[str] = []

    def _chk(label: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"  {PASS}  {label}")
        else:
            _failures.append(label)
            print(f"  {FAIL}  {label}" + (f"  [{detail}]" if detail else ""))

    def _section(title: str) -> None:
        print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

    # ── Smoke Test 1: Capgemini reference ─────────────────────────────────────
    _section("Smoke Test 1: Capgemini reference values (Results(8).xlsx)")

    capgemini = {
        "sig_foreign_hq_score":        3,
        "sig_explicit_lnd_score":      3,
        "sig_intl_footprint_score":    3,
        "sig_employer_branding_score": 2,
        "sig_lnd_onboarding_score":    2,
        "ti_onboarding_score":         2,
        "sig_rapid_growth_score":      1,
        "lusha_api_employee_range":    "100001 - 10000000",
    }
    r1 = score_company(capgemini)
    print(json.dumps({k: v for k, v in r1.items()
                      if k in ("lr_z_score", "lean_model_prob", "icp_similarity_score",
                               "company_size_score", "final_commercial_fit_score",
                               "commercial_tier", "model_probability")},
                     indent=2))

    _chk("lr_z_score ≈ 0.9870",
         abs(r1["lr_z_score"] - 0.9870) < 0.001,
         str(round(r1["lr_z_score"], 4)))
    _chk("lean_model_prob ≈ 0.7285",
         abs(r1["lean_model_prob"] - 0.7285) < 0.001,
         str(round(r1["lean_model_prob"], 4)))
    _chk("model_probability == lean_model_prob (alias, not separate calc)",
         r1["model_probability"] == r1["lean_model_prob"],
         str(r1["model_probability"]))
    _chk("model_probability ≠ 0.9992  (old wrong value)",
         abs(r1["model_probability"] - 0.9992) > 0.01,
         str(round(r1["model_probability"], 4)))
    _chk("icp_similarity_score ≈ 9.39",
         abs(r1["icp_similarity_score"] - 9.39) < 0.05,
         str(r1["icp_similarity_score"]))
    _chk("company_size_score = 10",
         r1["company_size_score"] == 10.0,
         str(r1["company_size_score"]))
    _chk("final_commercial_fit_score ≈ 9.54",
         abs(r1["final_commercial_fit_score"] - 9.54) < 0.05,
         str(r1["final_commercial_fit_score"]))
    _chk("final_commercial_fit_score ≠ 9.99  (old wrong value)",
         abs(r1["final_commercial_fit_score"] - 9.99) > 0.1,
         str(r1["final_commercial_fit_score"]))
    _chk("commercial_tier = 🥇 Hot",
         r1["commercial_tier"] == "🥇 Hot",
         r1["commercial_tier"])
    _chk("commercial_tier ≠ 'Tier 1'  (old wrong value)",
         r1["commercial_tier"] != "Tier 1",
         r1["commercial_tier"])
    _chk("company_size_missing is False",
         r1["company_size_missing"] is False)
    _chk("lean_model_logit == lr_z_score (alias)",
         r1["lean_model_logit"] == r1["lr_z_score"])

    # ── Smoke Test 2: All signals = 3, largest employee range ─────────────────
    _section("Smoke Test 2: Maximum signals, largest range")

    strong = {f: 3 for f in LEAN_COEFFICIENTS}
    strong["lusha_api_employee_range"] = "100001 - 10000000"
    r2 = score_company(strong)
    _chk("lean_model_prob > 0.65",  r2["lean_model_prob"] > 0.65, str(round(r2["lean_model_prob"], 4)))
    _chk("company_size_score = 10", r2["company_size_score"] == 10.0)
    _chk("commercial_tier = 🥇 Hot", r2["commercial_tier"] == "🥇 Hot", r2["commercial_tier"])
    _chk("high_value_flag is True",  r2["high_value_flag"] is True)

    # ── Smoke Test 3: All signals = 0, smallest range ─────────────────────────
    _section("Smoke Test 3: Minimum signals, smallest range")

    weak = {f: 0 for f in LEAN_COEFFICIENTS}
    weak["lusha_api_employee_range"] = "1 - 10"
    r3 = score_company(weak)
    _chk("lean_model_prob < 0.5",   r3["lean_model_prob"] < 0.5, str(round(r3["lean_model_prob"], 4)))
    _chk("company_size_score = 1.0", r3["company_size_score"] == 1.0)
    _chk("commercial_tier = ❄️ Pass", r3["commercial_tier"] == "❄️ Pass", r3["commercial_tier"])
    _chk("weak_flag is True",        r3["weak_flag"] is True)

    # ── Smoke Test 4: Missing employee range ──────────────────────────────────
    _section("Smoke Test 4: Missing employee range → defaults to 5.5")

    no_size = {f: 2 for f in LEAN_COEFFICIENTS}
    r4 = score_company(no_size)
    _chk("company_size_score = 5.5",    r4["company_size_score"] == SIZE_SCORE_MISSING,
         str(r4["company_size_score"]))
    _chk("company_size_missing is True", r4["company_size_missing"] is True)

    # ── Smoke Test 5: size band exact-string variants ─────────────────────────
    _section("Smoke Test 5: Employee range string variants")

    for raw, expected in [
        ("100001 - 10000000", 10.0),
        ("10001 - 100000",    8.88),
        ("5001 - 10000",      7.75),
        ("1001 - 5000",       6.63),
        ("501 - 1000",         5.5),
        ("201 - 500",          4.38),
        ("51 - 200",           3.25),
        ("11 - 50",            2.13),
        ("1 - 10",             1.0),
    ]:
        rz = score_company({"lusha_api_employee_range": raw})
        _chk(f"range '{raw}' → {expected}",
             rz["company_size_score"] == expected,
             str(rz["company_size_score"]))

    # ── Smoke Test 6: lusha_employee_range fallback ───────────────────────────
    _section("Smoke Test 6: lusha_employee_range fallback key")

    r6 = score_company({"lusha_employee_range": "1001 - 5000",
                         "sig_foreign_hq_score": 2})
    _chk("lusha_employee_range fallback → 6.63",
         r6["company_size_score"] == 6.63,
         str(r6["company_size_score"]))

    # ── Smoke Test 7: audit columns present ──────────────────────────────────
    _section("Smoke Test 7: All SCORE_OUTPUT_COLS produced by score_dataframe")

    test_df = pd.DataFrame([capgemini, strong, weak])
    out_df  = score_dataframe(test_df.copy())
    missing_cols = [c for c in SCORE_OUTPUT_COLS if c not in out_df.columns]
    _chk("all SCORE_OUTPUT_COLS present",
         len(missing_cols) == 0,
         str(missing_cols))
    _chk("row count unchanged", len(out_df) == 3)

    # ── Smoke Test 8: sig_merger_acq_score has NO effect ─────────────────────
    _section("Smoke Test 8: sig_merger_acq_score is NOT a coefficient input")

    row_with_merger  = {**capgemini, "sig_merger_acq_score": 3}
    row_no_merger    = {**capgemini, "sig_merger_acq_score": 0}
    r8a = score_company(row_with_merger)
    r8b = score_company(row_no_merger)
    _chk("sig_merger_acq_score=3 and =0 produce same lr_z_score",
         r8a["lr_z_score"] == r8b["lr_z_score"],
         f"{r8a['lr_z_score']} vs {r8b['lr_z_score']}")

    # ── Smoke Test 9: params override ────────────────────────────────────────
    _section("Smoke Test 9: intercept override drives prob to ~0")

    r9 = score_company(strong, params={"intercept": -99.0})
    _chk("intercept=-99 → lean_model_prob < 0.01",
         r9["lean_model_prob"] < 0.01, str(r9["lean_model_prob"]))
    _chk("intercept=-99 → ❄️ Pass",
         r9["commercial_tier"] == "❄️ Pass", r9["commercial_tier"])

    # ── Smoke Test 10: tier boundaries ───────────────────────────────────────
    _section("Smoke Test 10: Tier boundary values")

    for score_val, expected_tier in [
        (9.0,  "🥇 Hot"),
        (8.66, "🥇 Hot"),
        (8.0,  "🥈 Warm"),
        (7.19, "🥈 Warm"),
        (5.0,  "🥉 Cool"),
        (4.23, "🥉 Cool"),
        (2.0,  "❄️ Pass"),
    ]:
        row_t = {f: 0 for f in LEAN_COEFFICIENTS}
        row_t["lusha_api_employee_range"] = "1 - 10"
        r_t = score_company(row_t)
        # Directly test tier threshold logic
        computed_tier = TIER_THRESHOLDS[-1][1]
        for thresh, lbl in TIER_THRESHOLDS:
            if score_val >= thresh:
                computed_tier = lbl
                break
        _chk(f"score {score_val} → {expected_tier}",
             computed_tier == expected_tier,
             computed_tier)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    if _failures:
        print(f"  FAILURES ({len(_failures)}):")
        for f in _failures:
            print(f"    • {f}")
    else:
        print("  All smoke tests passed.")
    print("═"*60)
