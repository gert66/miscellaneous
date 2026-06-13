"""
Step 1: Offline seed-filter for the German OffeneRegister SQLite database.

Reads active German commercial company records, scores and labels them,
and exports filtered seed files for the mYngle B2B pipeline.

Output folder: C:/Users/gertm/Nextcloud/Myngle/Germany/01_seed/
(override with env var GERMANY_OUT_DIR; DB path via GERMANY_DB_PATH)
"""

import os
import re
import sqlite3
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — easy to adjust
# ---------------------------------------------------------------------------

_DEFAULT_DB  = Path(r"C:\Users\gertm\Nextcloud\Myngle\Germany\handelsregister.db")
_DEFAULT_OUT = Path(r"C:\Users\gertm\Nextcloud\Myngle\Germany\01_seed")

def _resolve_db_path() -> Path:
    env = os.environ.get("GERMANY_DB_PATH")
    if env:
        return Path(env)
    return _DEFAULT_DB

def _resolve_output_dir() -> Path:
    env = os.environ.get("GERMANY_OUT_DIR")
    if env:
        return Path(env)
    return _DEFAULT_OUT

DB_PATH    = _resolve_db_path()
OUTPUT_DIR = _resolve_output_dir()

SQL_LIMIT = 150_000      # rows to pull from the database
FINAL_LIMIT = 50_000     # rows to include in the main output after filtering

PRE_KEEP_THRESHOLD = 3   # score >= this → PRE_KEEP
PRE_MAYBE_THRESHOLD = 1  # score >= this → PRE_MAYBE (else PRE_UNKNOWN)
REVIEW_SAMPLE_N = 100    # rows per label in review sample

# ---------------------------------------------------------------------------
# Legal forms — order matters: longer/more specific patterns first
# ---------------------------------------------------------------------------

LEGAL_FORMS: list[tuple[str, str]] = [
    ("GmbH & Co. KGaA",  r"\bGmbH\s*&\s*Co\.?\s*KGaA\b"),
    ("GmbH & Co. KG",    r"\bGmbH\s*&\s*Co\.?\s*KG\b"),
    ("KGaA",             r"\bKGaA\b"),
    ("SE",               r"\bSE\b"),
    ("AG",               r"\bAG\b"),
    ("gGmbH",            r"\bgGmbH\b"),
    ("UG",               r"\bUG\b(?:\s*\(haftungsbeschränkt\))?"),
    ("GmbH",             r"\bGmbH\b"),
    ("KG",               r"\bKG\b"),
    ("OHG",              r"\bOHG\b"),
    ("e.K.",             r"\be\.K\.\b"),
    ("e.V.",             r"\be\.V\.\b"),
    ("Stiftung",         r"\bStiftung\b"),
    ("GbR",              r"\bGbR\b"),
]

# ---------------------------------------------------------------------------
# Scoring / exclusion rules
# ---------------------------------------------------------------------------

# Hard-exclude legal forms (no further scoring)
HARD_EXCLUDE_LEGAL_FORMS = {"e.V.", "Stiftung", "gGmbH", "UG", "e.K.", "GbR"}

# Strong positive legal forms
STRONG_POSITIVE_LEGAL_FORMS = {"AG", "SE", "KGaA", "GmbH & Co. KGaA", "GmbH & Co. KG"}
MODERATE_POSITIVE_LEGAL_FORMS = {"GmbH"}
SMALL_POSITIVE_LEGAL_FORMS = {"KG", "OHG"}

# Scale / prestige keywords (word-boundary aware)
SCALE_KEYWORDS: list[tuple[str, str]] = [
    ("gruppe",          r"\b(?:Gruppe|Group)\b"),
    ("holding",         r"\bHolding\b"),
    ("werke",           r"\bWerke\b"),
    ("international",   r"\b(?:International|Global)\b"),
    ("europe",          r"\b(?:Europe|Europa)\b"),
]

# Sector keywords
SECTOR_KEYWORDS: list[tuple[str, str]] = [
    ("maschinenbau",    r"\bMaschinenbau\b"),
    ("industrie",       r"\bIndustrie\b"),
    ("logistik",        r"\bLogistik\b"),
    ("automotive",      r"\bAutomotive\b"),
    ("technik",         r"\bTechnik\b"),
    ("technologie",     r"\bTechnologie(?:n)?\b"),
    ("systems",         r"\bSystems\b"),
    ("solutions",       r"\bSolutions\b"),
    ("software",        r"\bSoftware\b"),
    ("pharma",          r"\bPharma\b"),
    ("chemie",          r"\bChemie\b"),
    ("medtech",         r"\bMedTech\b"),
    ("engineering",     r"\bEngineering\b"),
    ("automation",      r"\bAutomation\b"),
    ("elektronik",      r"\bElektronik\b"),
    ("kunststoff",      r"\bKunststoff\b"),
    ("verpackung",      r"\b(?:Verpackung|Packaging)\b"),
    ("metall",          r"\bMetall\b"),
    ("medical",         r"\bMedical\b"),
    ("labor",           r"\bLabor\b"),
    ("energie",         r"\bEnergie\b"),
]

# Hard-exclude small/local business keywords
LOCAL_SMALL_KEYWORDS: list[tuple[str, str]] = [
    ("bäckerei",        r"\b(?:Bäckerei|Baeckerei)\b"),
    ("friseur",         r"\bFriseur\b"),
    ("metzgerei",       r"\b(?:Metzgerei|Fleischerei)\b"),
    ("restaurant",      r"\bRestaurant\b"),
    ("café",            r"\b(?:Café|Cafe)\b"),
    ("imbiss",          r"\bImbiss\b"),
    ("pizzeria",        r"\bPizzeria\b"),
    ("fahrrad",         r"\b(?:Fahrrad|Zweirad)\b"),
    ("fahrschule",      r"\bFahrschule\b"),
    ("zahnarzt",        r"\bZahnarzt\b"),
    ("arztpraxis",      r"\bArztpraxis\b"),
    ("praxis",          r"\bPraxis\b"),
    ("apotheke",        r"\bApotheke\b"),
    ("kosmetik",        r"\bKosmetik\b"),
    ("nagelstudio",     r"\bNagelstudio\b"),
    ("physio",          r"\bPhysio(?:therapie)?\b"),
    ("ergotherapie",    r"\bErgotherapie\b"),
]

# Public / non-commercial keywords — careful word-boundary matching
PUBLIC_KEYWORDS: list[tuple[str, str]] = [
    ("schule",          r"\b(?:Schule|Grundschule|Realschule|Gesamtschule|Berufsschule|Förderschule|Volksschule)\b"),
    ("universität",     r"\b(?:Universität|Universitaet|Hochschule|FH\b)"),
    ("kindergarten",    r"\b(?:Kindergarten|Kita)\b"),
    ("stadt",           r"\b(?:Stadt|Stadtgemeinde|Stadtrat|Stadtverwaltung|Stadtwerke)\b"),
    ("gemeinde",        r"\bGemeinde\b"),
    ("landkreis",       r"\bLandkreis\b"),
    ("kirche",          r"\b(?:Kirche|Pfarr(?:gemeinde|amt)?)\b"),
    ("verein",          r"\bVerein\b(?!\s*(?:igte|s))"),
    ("ihk",             r"\bIHK\b"),
    ("handwerkskammer", r"\bHandwerkskammer\b"),
    ("kammer",          r"\bKammer\b(?!er\b)"),
    ("amt",             r"\bAmt\b"),
]

# Low-priority keywords (reduce score, no hard exclusion)
LOW_PRIORITY_KEYWORDS: list[tuple[str, str]] = [
    ("grundstück",              r"\b(?:Grundstück|Grundstueck)\b"),
    ("grundbesitz",             r"\bGrundbesitz\b"),
    ("immobilien",              r"\bImmobilien\b"),
    ("real_estate",             r"\bReal\s*Estate\b"),
    ("property",                r"\bProperty\b"),
    ("objektgesellschaft",      r"\bObjektgesellschaft\b"),
    ("besitzgesellschaft",      r"\bBesitzgesellschaft\b"),
    ("verwaltungsgesellschaft", r"\bVerwaltungsgesellschaft\b"),
    ("vermögensverwaltung",     r"\b(?:Vermögensverwaltung|Vermoegensverwaltung)\b"),
    ("beteiligungsgesellschaft", r"\bBeteiligungsgesellschaft\b"),
    ("verwaltungs",             r"\bVerwaltungs\b"),
    ("residence",               r"\bResidence\b"),
    ("fonds",                   r"\bFonds\b"),
    ("asset_management",        r"\bAsset\s*Management\b"),
]

# Strong operational sector keywords — these override low-priority signals for PRE_KEEP
STRONG_OPERATIONAL_SECTOR_KEYWORDS = {
    "maschinenbau", "industrie", "logistik", "automotive", "technik",
    "technologie", "systems", "solutions", "software", "pharma", "chemie",
    "medtech", "engineering", "automation", "elektronik", "kunststoff",
    "verpackung", "metall", "medical", "energie",
}

# Compile all regex patterns once
def _compile(pairs: list[tuple[str, str]]) -> list[tuple[str, re.Pattern]]:
    return [(k, re.compile(p, re.IGNORECASE)) for k, p in pairs]


_LEGAL_FORM_COMPILED   = [(lf, re.compile(p)) for lf, p in LEGAL_FORMS]
_SCALE_COMPILED        = _compile(SCALE_KEYWORDS)
_SECTOR_COMPILED       = _compile(SECTOR_KEYWORDS)
_LOCAL_COMPILED        = _compile(LOCAL_SMALL_KEYWORDS)
_PUBLIC_COMPILED       = _compile(PUBLIC_KEYWORDS)
_LOW_PRI_COMPILED      = _compile(LOW_PRIORITY_KEYWORDS)

# ---------------------------------------------------------------------------
# Helper: clean company name
# ---------------------------------------------------------------------------

_EXTRA_WS = re.compile(r"\s{2,}")

def clean_name(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return _EXTRA_WS.sub(" ", raw.strip())


# ---------------------------------------------------------------------------
# Helper: detect legal form
# ---------------------------------------------------------------------------

def detect_legal_form(name: str) -> str:
    for lf, pat in _LEGAL_FORM_COMPILED:
        if pat.search(name):
            return lf
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Helper: score and label a single row
# ---------------------------------------------------------------------------

def score_row(name: str, legal_form: str) -> tuple[int, str, list[str], list[str], list[str]]:
    """Return (score, pre_label, positive_reasons, exclude_reasons, low_priority_reasons).

    PRE_KEEP rules (all must hold):
      - Strong legal form (AG/SE/KGaA/GmbH&Co.KG/GmbH&Co.KGaA) requires at least one
        sector or scale signal.
      - GmbH alone requires at least one sector or scale signal.
      - Low-priority signals block PRE_KEEP unless a strong operational sector keyword
        is present.
    """
    score = 0
    positive: list[str] = []
    excluded: list[str] = []
    low_pri: list[str] = []

    # --- Hard-exclude legal forms ---
    if legal_form in HARD_EXCLUDE_LEGAL_FORMS:
        excluded.append(f"excluded_legal_form:{legal_form}")
        return score, "PRE_EXCLUDE", positive, excluded, low_pri

    # --- Legal form scoring ---
    if legal_form in STRONG_POSITIVE_LEGAL_FORMS:
        score += 3
        positive.append(f"strong_legal_form:{legal_form}")
    elif legal_form in MODERATE_POSITIVE_LEGAL_FORMS:
        score += 2
        positive.append(f"kept_legal_form:{legal_form}")
    elif legal_form in SMALL_POSITIVE_LEGAL_FORMS:
        score += 1
        positive.append(f"kept_legal_form:{legal_form}")

    # --- Hard-exclude: local small business keywords ---
    for key, pat in _LOCAL_COMPILED:
        if pat.search(name):
            excluded.append(f"local_small_keyword:{key}")
            return score, "PRE_EXCLUDE", positive, excluded, low_pri

    # --- Hard-exclude: public / non-commercial keywords ---
    for key, pat in _PUBLIC_COMPILED:
        if pat.search(name):
            excluded.append(f"public_or_noncommercial_keyword:{key}")
            return score, "PRE_EXCLUDE", positive, excluded, low_pri

    # --- Scale keywords ---
    has_scale_signal = False
    for key, pat in _SCALE_COMPILED:
        if pat.search(name):
            score += 2
            has_scale_signal = True
            positive.append(f"positive_scale_keyword:{key}")

    # --- Sector keywords ---
    has_sector_signal = False
    has_strong_operational_signal = False
    for key, pat in _SECTOR_COMPILED:
        if pat.search(name):
            score += 1
            has_sector_signal = True
            if key in STRONG_OPERATIONAL_SECTOR_KEYWORDS:
                has_strong_operational_signal = True
            positive.append(f"positive_sector_keyword:{key}")

    # --- Low-priority keywords ---
    has_low_priority_signal = False
    for key, pat in _LOW_PRI_COMPILED:
        if pat.search(name):
            score -= 1
            has_low_priority_signal = True
            low_pri.append(f"low_priority_keyword:{key}")

    # --- PRE_KEEP qualification ---
    # A company qualifies for PRE_KEEP only when:
    #   1. Score is above threshold.
    #   2. There is at least one sector or scale signal (legal form alone is not enough).
    #   3. Low-priority signals do not block it, unless a strong operational sector
    #      keyword is present.
    has_positive_signal = has_sector_signal or has_scale_signal
    low_pri_blocks_keep = has_low_priority_signal and not has_strong_operational_signal

    if score >= PRE_KEEP_THRESHOLD and has_positive_signal and not low_pri_blocks_keep:
        label = "PRE_KEEP"
    elif score >= PRE_MAYBE_THRESHOLD:
        label = "PRE_MAYBE"
    else:
        label = "PRE_UNKNOWN"

    return score, label, positive, excluded, low_pri


# ---------------------------------------------------------------------------
# SQL query builder
# ---------------------------------------------------------------------------

def build_sql(limit: int) -> str:
    legal_form_conditions = " OR ".join([
        "name LIKE '%GmbH%'",
        "name LIKE '%Aktiengesellschaft%'",
        "name LIKE '% AG%'",
        "name LIKE '% SE%'",
        "name LIKE '%KGaA%'",
        "name LIKE '% KG%'",
        "name LIKE '% OHG%'",
        "name LIKE '%Kommanditgesellschaft%'",
    ])
    return f"""
        SELECT
            id,
            company_number,
            current_status,
            jurisdiction_code,
            name,
            registered_address,
            retrieved_at,
            federal_state,
            native_company_number,
            registered_office,
            registrar,
            register_art,
            register_nummer
        FROM company
        WHERE current_status = 'currently registered'
          AND jurisdiction_code = 'de'
          AND ({legal_form_conditions})
        LIMIT {limit}
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Resolved DB path : {DB_PATH}")
    print(f"Resolved out dir : {OUTPUT_DIR}")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {DB_PATH} ...")
    uri = DB_PATH.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)

    sql = build_sql(SQL_LIMIT)
    print(f"Loading up to {SQL_LIMIT:,} rows ...")
    df = pd.read_sql_query(sql, con)
    con.close()

    print(f"Rows loaded: {len(df):,}")

    # --- Rename / create output columns ---
    df = df.rename(columns={
        "id":                  "source_row_id",
        "name":                "company_name_raw",
        "registered_office":   "city_or_registered_office",
    })

    df["source"]             = "handelsregister.db"
    df["company_name_clean"] = df["company_name_raw"].apply(clean_name)
    df["legal_form_detected"] = df["company_name_clean"].apply(detect_legal_form)

    # --- Score each row ---
    results = df["company_name_clean"].apply(
        lambda n: score_row(n, detect_legal_form(n))
    )
    df["pre_score"]          = results.apply(lambda r: r[0])
    df["pre_label"]          = results.apply(lambda r: r[1])
    df["positive_reasons"]   = results.apply(lambda r: "; ".join(r[2]))
    df["exclude_reasons"]    = results.apply(lambda r: "; ".join(r[3]))
    df["low_priority_reasons"] = results.apply(lambda r: "; ".join(r[4]))

    # --- Select and order final columns ---
    OUTPUT_COLS = [
        "source", "source_row_id", "company_number", "current_status",
        "jurisdiction_code", "company_name_raw", "company_name_clean",
        "legal_form_detected", "registered_address", "city_or_registered_office",
        "federal_state", "native_company_number", "registrar", "register_art",
        "register_nummer", "retrieved_at", "pre_score", "pre_label",
        "positive_reasons", "exclude_reasons", "low_priority_reasons",
    ]
    df = df[OUTPUT_COLS]

    # --- Apply FINAL_LIMIT (keep non-excluded rows first) ---
    non_excluded = df[df["pre_label"] != "PRE_EXCLUDE"]
    excluded     = df[df["pre_label"] == "PRE_EXCLUDE"]

    if len(non_excluded) > FINAL_LIMIT:
        # Prioritise: PRE_KEEP > PRE_MAYBE > PRE_UNKNOWN
        non_excluded = non_excluded.sort_values(
            "pre_score", ascending=False
        ).head(FINAL_LIMIT)

    df_out = pd.concat([non_excluded, excluded], ignore_index=True)

    # --- Write main outputs ---
    csv_path  = OUTPUT_DIR / "germany_step1_seed_filtered.csv"
    xlsx_path = OUTPUT_DIR / "germany_step1_seed_filtered.xlsx"

    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_out.to_excel(xlsx_path, index=False, engine="openpyxl")

    print(f"Main CSV  : {csv_path}")
    print(f"Main XLSX : {xlsx_path}")

    # --- Counts file ---
    counts = (
        df_out.groupby(["pre_label", "legal_form_detected"])
        .size()
        .reset_index(name="count")
        .sort_values(["pre_label", "count"], ascending=[True, False])
    )
    counts_path = OUTPUT_DIR / "germany_step1_counts.csv"
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    print(f"Counts CSV: {counts_path}")

    # --- Review sample ---
    random.seed(42)
    sample_frames = []
    for label in ["PRE_KEEP", "PRE_MAYBE", "PRE_UNKNOWN", "PRE_EXCLUDE"]:
        subset = df_out[df_out["pre_label"] == label]
        n = min(REVIEW_SAMPLE_N, len(subset))
        if n:
            sample_frames.append(subset.sample(n, random_state=42))

    if sample_frames:
        review = pd.concat(sample_frames, ignore_index=True)
        review_path = OUTPUT_DIR / "germany_step1_review_sample.xlsx"
        review.to_excel(review_path, index=False, engine="openpyxl")
        print(f"Review XLSX: {review_path}")

    # --- Terminal summary ---
    label_counts = df_out["pre_label"].value_counts()
    print("\n--- Label counts ---")
    for label, cnt in label_counts.items():
        print(f"  {label}: {cnt:,}")

    print("\n--- Top legal forms by label ---")
    for label in ["PRE_KEEP", "PRE_MAYBE", "PRE_UNKNOWN", "PRE_EXCLUDE"]:
        top = (
            df_out[df_out["pre_label"] == label]["legal_form_detected"]
            .value_counts()
            .head(5)
        )
        print(f"\n  {label}:")
        for lf, cnt in top.items():
            print(f"    {lf}: {cnt:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
