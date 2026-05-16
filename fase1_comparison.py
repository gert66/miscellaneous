"""
ELM Enhanced — Signal Comparison & Prospect Ranking
====================================================
Takes the enriched CSV from fase1_elm_enhanced.py and produces:
  1. signal_comparison.csv  — per-signal lift of customers vs prospects
  2. prospect_scores.csv    — similarity score for every prospect row
  3. ranked_prospects.csv   — prospects sorted by similarity score

Usage:
  python fase1_comparison.py --input enriched_YYYYMMDD_HHMM.csv
  python fase1_comparison.py --input enriched.csv --output ./results
"""

import argparse
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Signal field definitions
# ─────────────────────────────────────────────────────────────────────────────

SCORE_FIELDS = [
    "elm_score_international",
    "elm_score_language_complexity",
    "elm_score_hiring",
    "elm_score_growth",
    "elm_score_training",
    "elm_score_workforce",
    "elm_score_technology",
    "elm_score_overall_icp",
]

BINARY_FIELDS = [
    "elm_hreflang_count",
    "elm_kw_international",
    "elm_kw_languages",
    "elm_kw_hiring",
    "elm_kw_growth",
    "elm_kw_training",
    "elm_kw_workforce",
    "elm_kw_technology",
    "elm_meta_description_found",
    "elm_og_tags_found",
    "elm_linkedin_link_found",
    "elm_careers_link_found",
    "elm_sitemap_found",
]

ALL_SIGNAL_FIELDS = SCORE_FIELDS + BINARY_FIELDS

_NAME_HINTS = ["company", "account", "name", "naam", "bedrijf"]
_URL_HINTS  = ["domain", "website", "url", "web", "site"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame):
    """Return (name_col, url_col) by fuzzy-matching column headers."""
    def best_match(hints, columns):
        best_col, best_ratio = None, 0.0
        for col in columns:
            col_lower = col.lower()
            for hint in hints:
                ratio = SequenceMatcher(None, hint, col_lower).ratio()
                if ratio > best_ratio:
                    best_ratio, best_col = ratio, col
        return best_col if best_ratio >= 0.45 else None

    cols = list(df.columns)
    return best_match(_NAME_HINTS, cols), best_match(_URL_HINTS, cols)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce booleans and strings to numeric; non-parseable → 0."""
    s = series.replace({True: 1, False: 0})
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _present(series: pd.Series) -> pd.Series:
    """Return boolean Series: True where value > 0."""
    return _coerce_numeric(series) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Function 1 — comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-signal comparison of customers vs prospects.
    Returns one row per signal with lift and strength label.
    """
    cust = df[df["list_type"] == "customer"]
    pros = df[df["list_type"] == "prospect"]

    n_cust = len(cust)
    n_pros = len(pros)

    if n_cust == 0 or n_pros == 0:
        raise ValueError("Need at least one customer and one prospect row.")

    rows = []
    for field in ALL_SIGNAL_FIELDS:
        if field not in df.columns:
            continue

        pct_c = round(_present(cust[field]).mean() * 100, 1) if n_cust else 0.0
        pct_p = round(_present(pros[field]).mean() * 100, 1) if n_pros else 0.0

        if pct_c < 5:
            continue

        lift = round(pct_c / pct_p, 2) if pct_p > 0 else None

        mean_c = round(_coerce_numeric(cust[field]).mean(), 2)
        mean_p = round(_coerce_numeric(pros[field]).mean(), 2)

        if lift is None:
            strength = "strong"
        elif lift >= 2.0:
            strength = "strong"
        elif lift >= 1.3:
            strength = "moderate"
        elif lift >= 1.0:
            strength = "weak"
        else:
            strength = "not distinctive"

        rows.append({
            "field":            field,
            "pct_customers":    pct_c,
            "pct_prospects":    pct_p,
            "diff_pp":          round(pct_c - pct_p, 1),
            "lift":             lift,
            "mean_customers":   mean_c,
            "mean_prospects":   mean_p,
            "signal_strength":  strength,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Sort by lift descending, nulls last
    result = result.sort_values("lift", ascending=False, na_position="last")
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Function 2 — similarity scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_scores(
    df: pd.DataFrame,
    comparison_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute customer_similarity_score (0-100) for each prospect row.
    Also adds top_signals and weak_signals columns.
    """
    prospects = df[df["list_type"] == "prospect"].copy()

    strong_fields   = set(comparison_table.loc[comparison_table["signal_strength"] == "strong",   "field"])
    moderate_fields = set(comparison_table.loc[comparison_table["signal_strength"] == "moderate", "field"])
    weighted_fields = list(strong_fields | moderate_fields)

    if not weighted_fields:
        prospects["customer_similarity_score"] = 0.0
        prospects["top_signals"]  = ""
        prospects["weak_signals"] = ""
        return prospects

    def weight(f):
        return 2 if f in strong_fields else 1

    total_weight = sum(weight(f) for f in weighted_fields)

    similarity_scores = []
    top_signals_list  = []
    weak_signals_list = []

    for _, row in prospects.iterrows():
        field_scores = {}
        for f in weighted_fields:
            raw = row.get(f, 0)
            # Normalise: score fields → /10, binary → bool 0/1
            if f in SCORE_FIELDS:
                norm = min(_coerce_numeric(pd.Series([raw])).iloc[0] / 10.0, 1.0)
            else:
                norm = 1.0 if _present(pd.Series([raw])).iloc[0] else 0.0
            field_scores[f] = norm

        weighted_sum = sum(field_scores[f] * weight(f) for f in weighted_fields)
        score = round((weighted_sum / total_weight) * 100, 1) if total_weight else 0.0
        similarity_scores.append(score)

        # Top 3 contributing fields (by weighted contribution, descending)
        contributions = sorted(
            weighted_fields,
            key=lambda f: field_scores[f] * weight(f),
            reverse=True,
        )
        top_signals_list.append(", ".join(contributions[:3]))

        # Fields where prospect scored 0
        weak = [f for f in weighted_fields if field_scores[f] == 0.0]
        weak_signals_list.append(", ".join(weak))

    prospects["customer_similarity_score"] = similarity_scores
    prospects["top_signals"]  = top_signals_list
    prospects["weak_signals"] = weak_signals_list
    return prospects


# ─────────────────────────────────────────────────────────────────────────────
# Function 3 — ranked prospect list
# ─────────────────────────────────────────────────────────────────────────────

def build_ranked_list(
    prospects_df: pd.DataFrame,
    name_col: str,
    url_col: str,
) -> pd.DataFrame:
    """Return prospects sorted by similarity score with a curated column set."""
    prospects_df = prospects_df.copy()

    prospects_df["website_data_quality"] = (
        prospects_df.get("elm_fetch_status", pd.Series("", index=prospects_df.index)) == "ok"
    )

    keep = [
        name_col,
        url_col,
        "elm_sector_cluster",
        "customer_similarity_score",
        "top_signals",
        "weak_signals",
        "elm_score_overall_icp",
        "elm_hreflang_count",
        "elm_score_language_complexity",
        "website_data_quality",
    ]
    # Only keep columns that actually exist
    keep = [c for c in keep if c in prospects_df.columns]

    return (
        prospects_df[keep]
        .sort_values("customer_similarity_score", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELM Enhanced — compare customer vs prospect signals and rank prospects"
    )
    parser.add_argument(
        "--input", required=True,
        help="Enriched CSV produced by fase1_elm_enhanced.py",
    )
    parser.add_argument(
        "--output", default="./output",
        help="Directory for output files (default: ./output)",
    )
    parser.add_argument(
        "--name-col", default=None,
        help="Column name for company name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--url-col", default=None,
        help="Column name for URL/domain (auto-detected if omitted)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Resolve name / url columns
    name_col = args.name_col
    url_col  = args.url_col
    if name_col is None or url_col is None:
        auto_name, auto_url = _detect_columns(df)
        name_col = name_col or auto_name
        url_col  = url_col  or auto_url

    if name_col is None:
        sys.exit("ERROR: could not detect a company-name column. Pass --name-col.")
    if url_col is None:
        sys.exit("ERROR: could not detect a URL column. Pass --url-col.")

    if "list_type" not in df.columns:
        sys.exit("ERROR: input CSV has no 'list_type' column. Run fase1_elm_enhanced.py first.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: comparison table ──────────────────────────────────────────────
    print("Building signal comparison table…")
    comparison = build_comparison_table(df)
    comp_path = out_dir / "signal_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"  → {comp_path}  ({len(comparison)} signals)")

    strong_count   = (comparison["signal_strength"] == "strong").sum()
    moderate_count = (comparison["signal_strength"] == "moderate").sum()
    print(f"     strong: {strong_count}   moderate: {moderate_count}")

    # ── Step 2: similarity scores ─────────────────────────────────────────────
    print("Computing prospect similarity scores…")
    scored_prospects = compute_similarity_scores(df, comparison)
    scores_path = out_dir / "prospect_scores.csv"
    scored_prospects.to_csv(scores_path, index=False)
    print(f"  → {scores_path}  ({len(scored_prospects)} prospects)")

    # ── Step 3: ranked list ───────────────────────────────────────────────────
    print("Building ranked prospect list…")
    ranked = build_ranked_list(scored_prospects, name_col, url_col)
    ranked_path = out_dir / "ranked_prospects.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"  → {ranked_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_cust = (df["list_type"] == "customer").sum()
    n_pros = (df["list_type"] == "prospect").sum()
    top5 = ranked.head(5)

    print(f"\n{'='*55}")
    print(f"  Customers analysed : {n_cust}")
    print(f"  Prospects scored   : {n_pros}")
    print(f"  Distinctive signals: {strong_count} strong, {moderate_count} moderate")
    print(f"\n  Top 5 prospects by similarity:")
    for _, row in top5.iterrows():
        company = row.get(name_col, "—")
        score   = row.get("customer_similarity_score", 0)
        sector  = row.get("elm_sector_cluster", "—")
        print(f"    {company[:40]:<40}  {score:>5.1f}  [{sector}]")
    print(f"{'='*55}")
