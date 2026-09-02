"""03b_extract_full_json_materials.py

Creates the pipeline-ready material dataset from the JSON material-evidence
parser (json_material_evidence_audit.py / material_parser_core.py).

Input:
    data/interim/garments_clean.csv

Outputs:
    data/interim/garments_with_full_materials.csv
    data/interim/material_audit_full_json.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS, PRIMARY_FOLDER, RAW_DATA
from src.json_material_evidence_audit import inspect_json_record, normalise_id
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = INTERIM_DATA / "garments_clean.csv"
JSON_DIRECTORY = RAW_DATA / PRIMARY_FOLDER / "styles"

OUTPUT_FILE = INTERIM_DATA / "garments_with_full_materials.csv"
AUDIT_FILE = INTERIM_DATA / "material_audit_full_json.csv"

AUDIT_COLUMNS = [
    "garment_id", "product_name", "gender", "article_type",
    "primary_material", "secondary_materials", "material_variants",
    "composition_percentages_json", "material_role_flags", "material_source",
    "material_confidence", "material_evidence", "composition_status",
    "scoring_eligibility", "exclusion_reason", "source_conflict",
    "source_conflict_detail", "json_error",
]

DUPLICATE_COLUMNS = {
    "garment_id", "csv_gender", "csv_master_category", "csv_sub_category",
    "csv_article_type", "csv_product_name", "json_age_group", "json_gender",
}


def missing_json_result(garment_id: str, json_path: Path) -> dict[str, Any]:
    """Default record when no JSON file exists for a garment."""
    return {
        "garment_id": garment_id,
        "json_file_found": False,
        "json_valid": False,
        "json_path": str(json_path),
        "json_id": "",
        "json_id_matches_garment_id": False,
        "json_error": "json_file_not_found",
        "extracted_material": "",
        "primary_material": "",
        "secondary_materials": "",
        "material_variants": "",
        "composition_percentages_json": "[]",
        "component_count": 0,
        "percentage_total": pd.NA,
        "percentage_total_status": "not_available",
        "material_role_flags": "",
        "composition_status": "unknown",
        "material_evidence_confidence": "none",
        "material_source": "unknown",
        "material_confidence": "none",
        "material_status": "unknown",
        "material_evidence": "",
        "scoring_eligibility": "not_eligible_missing_material_evidence",
        "exclusion_reason": "json_file_not_found",
        "elastane_present": False,
        "recycled_content_claim": False,
        "source_conflict": False,
        "source_conflict_detail": "",
    }


def extract_full_json_materials() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\nRun 02_clean_apparel_metadata.py first."
        )
    if not JSON_DIRECTORY.exists():
        raise FileNotFoundError(f"JSON directory not found: {JSON_DIRECTORY}")

    INTERIM_DATA.mkdir(parents=True, exist_ok=True)

    garments = pd.read_csv(INPUT_FILE, low_memory=False)
    if "garment_id" not in garments.columns:
        raise KeyError("Input dataset must contain 'garment_id'.")

    garments["garment_id"] = garments["garment_id"].apply(normalise_id)

    enriched_records: list[dict[str, Any]] = []

    for _, row in garments.iterrows():
        garment_id = row["garment_id"]
        json_path = JSON_DIRECTORY / f"{garment_id}.json"
        garment_base = row.to_dict()

        if not json_path.exists():
            enriched_records.append(
                {**garment_base, **missing_json_result(garment_id, json_path)}
            )
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            material_record, _ = inspect_json_record(
                garment_id=garment_id, payload=payload, json_path=json_path,
            )

            for column in DUPLICATE_COLUMNS:
                material_record.pop(column, None)

            enriched_records.append({**garment_base, **material_record})

        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
            error_result = missing_json_result(garment_id, json_path)
            error_result.update(
                {
                    "json_file_found": True,
                    "json_error": type(error).__name__,
                    "exclusion_reason": "invalid_or_unreadable_json",
                }
            )
            enriched_records.append({**garment_base, **error_result})

    output_df = pd.DataFrame(enriched_records)
    output_df.to_csv(OUTPUT_FILE, index=False)

    audit_mask = output_df["scoring_eligibility"].ne("eligible_exact_single_fibre")
    available_audit_columns = [c for c in AUDIT_COLUMNS if c in output_df.columns]
    output_df.loc[audit_mask, available_audit_columns].to_csv(AUDIT_FILE, index=False)

    logger.info("Full JSON material extraction completed.")
    logger.info("Total garments processed: %s", f"{len(output_df):,}")
    logger.info("Output saved to: %s", OUTPUT_FILE)
    logger.info("Audit file saved to: %s", AUDIT_FILE)
    logger.info(
        "Composition status breakdown:\n%s",
        output_df["composition_status"].value_counts(dropna=False).to_string(),
    )
    logger.info(
        "Scoring eligibility breakdown:\n%s",
        output_df["scoring_eligibility"].value_counts(dropna=False).to_string(),
    )


if __name__ == "__main__":
    extract_full_json_materials()