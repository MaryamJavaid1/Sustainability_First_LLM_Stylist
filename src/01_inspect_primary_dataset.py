"""01_inspect_primary_dataset.py

Diagnostic step only. Locates and inspects the raw Kaggle metadata file
(styles.csv) to check basic integrity before any cleaning happens.

This script does not clean, filter, score, or otherwise modify the dataset.
It uses ``on_bad_lines="skip"`` for a quick read, so malformed rows are
skipped here rather than repaired row repair happens later, and is logged,
in 02_clean_apparel_metadata.py.

Input:
    data/raw/fashion_product_images_full/styles.csv

Output:
    outputs/primary_dataset_inspection.txt
"""

from __future__ import annotations

import pandas as pd

from src.config import OUTPUTS, PRIMARY_FOLDER, RAW_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)


def inspect_primary_dataset() -> None:
    """Locate and inspect the raw Kaggle metadata file."""
    target_dir = RAW_DATA / PRIMARY_FOLDER
    styles_files = list(target_dir.rglob("styles.csv"))

    if not styles_files:
        raise FileNotFoundError(f"Could not find styles.csv in {target_dir}")

    styles_path = styles_files[0]

    # low_memory=False avoids pandas guessing dtypes across mixed-type columns.
    df = pd.read_csv(styles_path, on_bad_lines="skip", low_memory=False)

    report_path = OUTPUTS / "primary_dataset_inspection.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"Source file: {styles_path}\n")
        report.write(f"Rows: {df.shape[0]}\n")
        report.write(f"Columns: {df.shape[1]}\n\n")
        report.write("Column names:\n")
        report.write("\n".join(df.columns.astype(str).tolist()))
        report.write("\n\nMissing values:\n")
        report.write(df.isna().sum().sort_values(ascending=False).to_string())

    logger.info("Inspection complete. Report saved to: %s", report_path)


if __name__ == "__main__":
    inspect_primary_dataset()