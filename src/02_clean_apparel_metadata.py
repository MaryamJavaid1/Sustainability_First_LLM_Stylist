"""02_clean_apparel_metadata.py

Creates a clean, adult-only Apparel dataset for the sustainability-first
wardrobe project.

Adult scope:
    - Adults-Men
    - Adults-Women
    - Adults-Unisex

Excluded:
    - Kids-Boys / Kids-Girls / Kids-Unisex
    - Missing, invalid, or ambiguous JSON ageGroup records
    - Non-Apparel records

Inputs:
    data/raw/fashion_product_images_full/styles.csv
    data/raw/fashion_product_images_full/styles/<garment_id>.json

Outputs:
    data/interim/garments_clean.csv
    data/interim/adult_scope_excluded.csv
    data/interim/styles_csv_parse_repairs.csv
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS, PRIMARY_FOLDER, RAW_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = RAW_DATA / PRIMARY_FOLDER / "styles.csv"
JSON_DIRECTORY = RAW_DATA / PRIMARY_FOLDER / "styles"

OUTPUT_FILE = INTERIM_DATA / "garments_clean.csv"
EXCLUDED_FILE = INTERIM_DATA / "adult_scope_excluded.csv"
CSV_REPAIR_AUDIT_FILE = INTERIM_DATA / "styles_csv_parse_repairs.csv"

COLUMN_RENAME_MAP = {
    "id": "garment_id",
    "masterCategory": "master_category",
    "subCategory": "sub_category",
    "articleType": "article_type",
    "baseColour": "base_colour",
    "productDisplayName": "product_name",
}

CHILD_AGE_TERMS = {"boy", "girl", "kid", "child", "children", "baby", "infant"}


def normalise_garment_id(value: Any) -> str:
    """Convert values such as 1163.0 into '1163', matching 1163.json."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def read_styles_csv_with_repair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read styles.csv while preserving and logging malformed rows.

    The expected file has ten columns and productDisplayName is the final
    column. If an unquoted comma creates extra fields, the overflow is merged
    back into productDisplayName and logged in a repair audit file. Rows with
    too few fields are padded with empty values and also logged.
    """
    repaired_rows: list[list[str]] = []
    repair_log: list[dict[str, Any]] = []

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Source CSV is empty: {INPUT_FILE}") from error

        expected_field_count = len(header)

        if expected_field_count != 10:
            raise ValueError(
                f"Expected 10 columns in styles.csv, but found "
                f"{expected_field_count}: {header}"
            )

        if header[-1] != "productDisplayName":
            raise ValueError(
                "Expected productDisplayName as final CSV column, "
                f"but found: {header[-1]}"
            )

        for source_line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue

            original_field_count = len(row)

            if original_field_count == expected_field_count:
                repaired_rows.append(row)
                continue

            if original_field_count > expected_field_count:
                repaired_row = row[: expected_field_count - 1] + [
                    ",".join(row[expected_field_count - 1 :])
                ]
                action = "merged_extra_fields_into_productDisplayName"
            else:
                repaired_row = row + [""] * (expected_field_count - original_field_count)
                action = "padded_missing_fields_with_empty_values"

            repaired_rows.append(repaired_row)
            repair_log.append(
                {
                    "source_line_number": source_line_number,
                    "garment_id": normalise_garment_id(repaired_row[0]),
                    "original_field_count": original_field_count,
                    "expected_field_count": expected_field_count,
                    "repair_action": action,
                    "reconstructed_product_display_name": repaired_row[-1],
                }
            )

    raw_df = pd.DataFrame(repaired_rows, columns=header)

    repair_audit = pd.DataFrame(
        repair_log,
        columns=[
            "source_line_number",
            "garment_id",
            "original_field_count",
            "expected_field_count",
            "repair_action",
            "reconstructed_product_display_name",
        ],
    )
    repair_audit.to_csv(CSV_REPAIR_AUDIT_FILE, index=False)

    return raw_df, repair_audit


def get_json_age_group(garment_id: str) -> tuple[str, str, bool, str]:
    """Read ageGroup and gender from a product JSON file.

    Returns: age_group, json_gender, json_file_found, json_error
    """
    if not garment_id:
        return "", "", False, "missing_garment_id"

    json_path = JSON_DIRECTORY / f"{garment_id}.json"
    if not json_path.exists():
        return "", "", False, "json_file_not_found"

    try:
        with json_path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        data = record.get("data", {})
        if not isinstance(data, dict):
            return "", "", True, "invalid_json_data"

        age_group = str(data.get("ageGroup", "")).strip()
        json_gender = str(data.get("gender", "")).strip()
        return age_group, json_gender, True, ""

    except json.JSONDecodeError:
        return "", "", True, "json_decode_error"
    except (OSError, UnicodeDecodeError):
        return "", "", True, "json_read_error"


def classify_adult_scope(garment_id: str) -> dict[str, Any]:
    """Classify an Apparel record's adult scope using JSON ageGroup only.

    Only JSON-confirmed adult records are retained; this avoids relying on
    incomplete or ambiguous CSV gender labels.
    """
    age_group, json_gender, json_file_found, json_error = get_json_age_group(garment_id)
    age_group_clean = age_group.casefold()

    result = {
        "json_age_group": age_group,
        "json_gender": json_gender,
        "json_file_found": json_file_found,
        "json_error": json_error,
        "adult_scope_status": "excluded",
        "adult_scope_reason": "unverified_json_age_group",
    }

    if age_group_clean.startswith("adults-"):
        result["adult_scope_status"] = "included"
        result["adult_scope_reason"] = "verified_adult_from_json_age_group"
    elif any(term in age_group_clean for term in CHILD_AGE_TERMS):
        result["adult_scope_reason"] = "verified_child_from_json_age_group"
    elif not json_file_found:
        result["adult_scope_reason"] = "json_file_not_found"
    elif json_error:
        result["adult_scope_reason"] = json_error
    elif not age_group_clean:
        result["adult_scope_reason"] = "missing_json_age_group"
    else:
        result["adult_scope_reason"] = "ambiguous_or_unrecognised_json_age_group"

    return result


def clean_adult_apparel_metadata() -> None:
    """Create the clean, adult-only Apparel metadata dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Raw source file not found: {INPUT_FILE}")
    if not JSON_DIRECTORY.exists():
        raise FileNotFoundError(f"JSON directory not found: {JSON_DIRECTORY}")

    INTERIM_DATA.mkdir(parents=True, exist_ok=True)

    raw_df, csv_repair_audit = read_styles_csv_with_repair()
    total_raw_records = len(raw_df)

    df = raw_df.rename(columns=COLUMN_RENAME_MAP).copy()

    required_columns = {
        "garment_id", "gender", "master_category", "sub_category",
        "article_type", "base_colour", "season", "year", "usage", "product_name",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing expected source columns: {sorted(missing_columns)}")

    df["source_dataset"] = "Fashion Product Images"
    df["source_file"] = INPUT_FILE.name
    df["record_status"] = "source_metadata"
    df["garment_id"] = df["garment_id"].apply(normalise_garment_id)

    df["master_category_clean"] = (
        df["master_category"].fillna("").astype(str).str.strip().str.casefold()
    )

    apparel_df = df.loc[df["master_category_clean"].eq("apparel")].copy()
    total_apparel_records = len(apparel_df)

    apparel_df = apparel_df.loc[apparel_df["garment_id"].ne("")].copy()
    apparel_df = apparel_df.drop_duplicates(subset=["garment_id"]).copy()

    apparel_df["gender_clean"] = (
        apparel_df["gender"].fillna("").astype(str).str.strip().str.casefold()
    )

    scope_columns = [
        "json_age_group", "json_gender", "json_file_found",
        "json_error", "adult_scope_status", "adult_scope_reason",
    ]
    for column in scope_columns:
        apparel_df[column] = ""
    apparel_df["json_file_found"] = False
    apparel_df["adult_scope_status"] = "excluded"
    apparel_df["adult_scope_reason"] = "unverified_json_age_group"

    for row_index, row in apparel_df.iterrows():
        scope_result = classify_adult_scope(row["garment_id"])
        for column, value in scope_result.items():
            apparel_df.at[row_index, column] = value

    adult_df = apparel_df.loc[apparel_df["adult_scope_status"].eq("included")].copy()
    excluded_df = apparel_df.loc[apparel_df["adult_scope_status"].ne("included")].copy()

    helper_columns = ["master_category_clean", "gender_clean"]
    adult_df = adult_df.drop(columns=helper_columns, errors="ignore")
    excluded_df = excluded_df.drop(columns=helper_columns, errors="ignore")

    adult_df.to_csv(OUTPUT_FILE, index=False)
    excluded_df.to_csv(EXCLUDED_FILE, index=False)

    logger.info("Adult-only Apparel cleaning completed.")
    logger.info("Raw dataset records: %s", f"{total_raw_records:,}")
    logger.info("CSV parsing repairs applied: %s", f"{len(csv_repair_audit):,}")
    logger.info("CSV repair audit saved to: %s", CSV_REPAIR_AUDIT_FILE)
    logger.info("Apparel records before adult filtering: %s", f"{total_apparel_records:,}")
    logger.info("Adult-only Apparel records retained: %s", f"{len(adult_df):,}")
    logger.info("Excluded Apparel records: %s", f"{len(excluded_df):,}")
    logger.info("Adult clean dataset saved to: %s", OUTPUT_FILE)
    logger.info("Adult-scope audit saved to: %s", EXCLUDED_FILE)

    logger.info(
        "Included adult-scope summary:\n%s",
        adult_df["adult_scope_reason"].value_counts(dropna=False).to_string(),
    )
    logger.info(
        "Excluded-scope summary:\n%s",
        excluded_df["adult_scope_reason"].value_counts(dropna=False).to_string(),
    )


if __name__ == "__main__":
    clean_adult_apparel_metadata()