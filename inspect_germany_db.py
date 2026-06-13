"""
Inspect the German OffeneRegister SQLite database in read-only mode.
Prints tables, columns, row counts and 5-row samples.
FTS shadow tables (large blob tables) are listed but not sampled.
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "Germany" / "handelsregister.db"

FTS_SHADOW_SUFFIXES = (
    "_content", "_segments", "_segdir", "_docsize", "_stat",
    "_data", "_idx", "_config",
)


def is_fts_shadow(table_name: str, all_tables: list[str]) -> bool:
    for suffix in FTS_SHADOW_SUFFIXES:
        if table_name.endswith(suffix):
            base = table_name[: -len(suffix)]
            if base in all_tables:
                return True
    return False


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    uri = DB_PATH.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name")
    objects = cur.fetchall()
    all_table_names = [r["name"] for r in objects if r["type"] == "table"]

    print(f"\nDatabase: {DB_PATH}")
    print(f"Tables/views found: {len(objects)}\n")

    for row in objects:
        name = row["name"]
        kind = row["type"].upper()

        if is_fts_shadow(name, all_table_names):
            print(f"  [{kind}] {name}  (FTS shadow — skipped)")
            continue

        try:
            cur.execute(f'SELECT COUNT(*) AS cnt FROM "{name}"')
            count = cur.fetchone()["cnt"]
        except sqlite3.OperationalError as e:
            print(f"  [{kind}] {name}  ERROR counting rows: {e}")
            continue

        cur.execute(f'PRAGMA table_info("{name}")')
        cols = [c["name"] for c in cur.fetchall()]

        print(f"\n{'='*60}")
        print(f"[{kind}] {name}")
        print(f"  Rows   : {count:,}")
        print(f"  Columns: {cols}")

        try:
            cur.execute(f'SELECT * FROM "{name}" LIMIT 5')
            samples = cur.fetchall()
            print("  Sample (up to 5 rows):")
            for i, s in enumerate(samples, 1):
                print(f"    [{i}] {dict(s)}")
        except sqlite3.OperationalError as e:
            print(f"  Could not fetch sample: {e}")

    con.close()
    print("\nInspection complete.")


if __name__ == "__main__":
    main()
