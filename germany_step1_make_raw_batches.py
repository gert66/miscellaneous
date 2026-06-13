"""
Split the Germany PRE_KEEP seed file into 500-row raw input batches.

Default input:
  C:/Users/gertm/Nextcloud/Myngle/Germany/01_seed/germany_step1_PRE_KEEP_for_serper.xlsx

Default output folder:
  C:/Users/gertm/Nextcloud/Myngle/Germany/00_raw/

Output filename pattern:
  Germany_1_R0001_0500.xlsx
  Germany_2_R0501_1000.xlsx
  ...

Run with defaults:
  python germany_step1_make_raw_batches.py

Override options:
  --input       path to input XLSX
  --output-dir  path to output folder
  --batch-size  rows per batch (default 500)
  --prefix      filename prefix (default Germany)
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT      = Path(r"C:\Users\gertm\Nextcloud\Myngle\Germany\01_seed\germany_step1_PRE_KEEP_for_serper.xlsx")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\gertm\Nextcloud\Myngle\Germany\00_raw")
DEFAULT_BATCH_SIZE = 500
DEFAULT_PREFIX     = "Germany"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split Germany PRE_KEEP file into raw batches.")
    p.add_argument("--input",       type=Path, default=DEFAULT_INPUT,
                   help="Input XLSX file (default: Germany PRE_KEEP for Serper)")
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Output folder for batch files")
    p.add_argument("--batch-size",  type=int,  default=DEFAULT_BATCH_SIZE,
                   help="Rows per batch (default 500)")
    p.add_argument("--prefix",      type=str,  default=DEFAULT_PREFIX,
                   help="Filename prefix (default Germany)")
    return p.parse_args()


def archive_existing(output_dir: Path) -> None:
    """Move existing Germany_*.xlsx and manifest to a timestamped archive folder."""
    existing_batches = sorted(output_dir.glob("Germany_*.xlsx"))
    existing_manifest = output_dir / "Germany_batch_manifest.csv"

    targets = existing_batches[:]
    if existing_manifest.exists():
        targets.append(existing_manifest)

    if not targets:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Archive lives one level above 00_raw, in Germany/_archive/
    archive_root = output_dir.parent / "_archive"
    archive_dir  = archive_root / f"raw_batches_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWARNING: {len(existing_batches)} existing batch file(s) found in {output_dir}")
    print(f"Archiving to: {archive_dir}")

    for f in targets:
        dest = archive_dir / f.name
        shutil.move(str(f), str(dest))
        print(f"  Archived: {f.name}")

    print()


def batch_filename(prefix: str, batch_num: int, row_start: int, row_end: int) -> str:
    return f"{prefix}_{batch_num}_R{row_start:04d}_{row_end:04d}.xlsx"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_file  = args.input
    output_dir  = args.output_dir
    batch_size  = args.batch_size
    prefix      = args.prefix

    print(f"Input file  : {input_file}")
    print(f"Output dir  : {output_dir}")
    print(f"Batch size  : {batch_size}")
    print(f"Prefix      : {prefix}")

    # --- Load input ---
    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        sys.exit(1)

    print(f"\nLoading {input_file.name} ...")
    df = pd.read_excel(input_file, engine="openpyxl")
    print(f"Rows loaded : {len(df):,}")

    if df.empty:
        print("ERROR: Input file is empty.")
        sys.exit(1)

    # --- Sort ---
    if "pre_score" not in df.columns:
        print("WARNING: pre_score column not found — keeping original order.")
    else:
        sort_cols = ["pre_score"]
        sort_asc  = [False]
        if "company_name_clean" in df.columns:
            sort_cols.append("company_name_clean")
            sort_asc.append(True)
        df = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    # --- Safety: archive any existing batch files ---
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_existing(output_dir)

    # --- Create batches ---
    total_rows = len(df)
    import math
    n_batches = math.ceil(total_rows / batch_size)

    manifest_rows = []
    first_file: Path | None = None
    last_file:  Path | None = None
    created_at = datetime.now().isoformat(timespec="seconds")

    print(f"Creating {n_batches} batch(es) ...")

    for i in range(n_batches):
        row_start = i * batch_size           # 0-based slice start
        row_end   = min(row_start + batch_size, total_rows)

        label_start = row_start + 1          # 1-based for filename
        label_end   = row_end

        batch_df   = df.iloc[row_start:row_end].reset_index(drop=True)
        batch_num  = i + 1
        fname      = batch_filename(prefix, batch_num, label_start, label_end)
        fpath      = output_dir / fname

        batch_df.to_excel(fpath, index=False, engine="openpyxl")
        print(f"  [{batch_num:>3}] {fname}  ({len(batch_df):,} rows)")

        manifest_rows.append({
            "batch_number": batch_num,
            "row_start":    label_start,
            "row_end":      label_end,
            "rows":         len(batch_df),
            "filename":     fname,
            "full_path":    str(fpath),
            "input_file":   str(input_file),
            "created_at":   created_at,
        })

        if first_file is None:
            first_file = fpath
        last_file = fpath

    # --- Write manifest ---
    manifest_path = output_dir / "Germany_batch_manifest.csv"
    manifest_df   = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    # --- Terminal summary ---
    print("\n" + "=" * 60)
    print(f"  Rows loaded       : {total_rows:,}")
    print(f"  Batches created   : {n_batches}")
    print(f"  Batch size        : {batch_size}")
    print(f"  First batch       : {first_file.name if first_file else '-'}")
    print(f"  Last batch        : {last_file.name if last_file else '-'}")
    print(f"  Manifest          : {manifest_path}")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
