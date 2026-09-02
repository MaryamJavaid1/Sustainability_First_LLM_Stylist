"""test_pipeline.py

Integration tests for the sustainability-stylist pipeline.

Unlike test_material_parser_core.py and test_json_material_evidence_audit.py,
which test individual parsing functions in isolation, this file runs the
actual pipeline STAGE SCRIPTS (02, 03b, 04, 05, 06) end-to-end against a
small synthetic dataset built in a pytest tmp_path. This is the test most
likely to catch a real bug: e.g. a column renamed in one stage that a later
stage still expects under its old name.

How it works:
    - `pipeline_paths` builds a fake data/raw/.../styles.csv plus a handful
      of per-garment JSON files in a temp directory.
    - `patched_stage_modules` imports each stage module (via importlib,
      since filenames like "02_clean_apparel_metadata.py" aren't valid
      Python identifiers you can `import` directly) and monkeypatches its
      module-level path constants (INPUT_FILE, OUTPUT_FILE, etc.) to point
      into the temp directory instead of your real data/ folder.
    - Each test then calls the stage's public function directly and checks
      the resulting CSV.

Note: because logging is configured once per process (see
src/logging_utils.py), running these tests may write a line or two to your
real outputs/pipeline_run.log. This is harmless it does not touch your
real data/ directory, only the log file.
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


# Import stage modules

# Filenames like "02_clean_apparel_metadata.py" cannot be imported with a
# literal `import src.02_clean_apparel_metadata` statement (not a valid
# Python identifier), so we use importlib.import_module with a string path
# instead this is the same mechanism `python -m src.02_...` relies on.
stage02 = importlib.import_module("src.02_clean_apparel_metadata")
stage03b = importlib.import_module("src.03b_extract_full_json_materials")
stage04 = importlib.import_module("src.04_apply_lca_scores")
stage05 = importlib.import_module("src.05_generate_wardrobe_attributes")
stage06 = importlib.import_module("src.06_calculate_final_sustainability")

CSV_HEADER = [
    "id", "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "year", "usage", "productDisplayName",
]

# Four synthetic garments covering the branches that matter most:
#   1001  adult men, single fibre (100% cotton)          -> should survive to the end
#   1002  adult women, exact blend (60/40 cotton/polyester) -> filtered out at scoring
#           (known_blend_pending_policy is intentionally left unscored see
#           Chapter 3, "Deviations from the Approved Proposal")
#   1003 kids apparel                                    -> excluded at stage 02
#   1004 adult footwear (non-Apparel)                     -> excluded at stage 02
CSV_ROWS = [
    ["1001", "Men", "Apparel", "Topwear", "Tshirts", "Black", "Summer", "2024", "Casual", "Test Cotton Tee"],
    ["1002", "Women", "Apparel", "Topwear", "Shirts", "Blue", "Winter", "2024", "Casual", "Test Blend Shirt"],
    ["1003", "Boys", "Apparel", "Topwear", "Tshirts", "Red", "Summer", "2024", "Casual", "Kids Tee"],
    ["1004", "Unisex", "Footwear", "Shoes", "Sneakers", "White", "Summer", "2024", "Casual", "Test Sneaker"],
]

JSON_PAYLOADS = {
    "1001": {
        "data": {
            "id": 1001,
            "ageGroup": "Adults-Men",
            "gender": "Men",
            "productDisplayName": "Test Cotton Tee",
            "productDescriptors": {
                "materials_care_desc": {"value": "<p>100% cotton.</p>"}
            },
            "articleAttributes": {},
        }
    },
    "1002": {
        "data": {
            "id": 1002,
            "ageGroup": "Adults-Women",
            "gender": "Women",
            "productDisplayName": "Test Blend Shirt",
            "productDescriptors": {
                "materials_care_desc": {
                    "value": "<p>60% cotton and 40% polyester.</p>"
                }
            },
            "articleAttributes": {},
        }
    },
    "1003": {
        "data": {
            "id": 1003,
            "ageGroup": "Kids-Boys",
            "gender": "Boys",
            "productDisplayName": "Kids Tee",
            "productDescriptors": {
                "materials_care_desc": {"value": "<p>100% cotton.</p>"}
            },
            "articleAttributes": {},
        }
    },
    # No JSON file for 1004 (Footwear) it's excluded by category before
    # stage 02 ever looks for a JSON file, so this deliberately tests that
    # a missing file for a non-Apparel record doesn't raise an error.
}


@pytest.fixture
def pipeline_paths(tmp_path: Path) -> dict[str, Path]:
    """Build a small synthetic raw dataset under a temp directory."""
    raw_dir = tmp_path / "data" / "raw" / "fashion_product_images_full"
    json_dir = raw_dir / "styles"
    interim_dir = tmp_path / "data" / "interim"
    processed_dir = tmp_path / "data" / "processed"

    json_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    styles_csv = raw_dir / "styles.csv"
    with styles_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(CSV_ROWS)

    for garment_id, payload in JSON_PAYLOADS.items():
        with (json_dir / f"{garment_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    return {
        "raw_dir": raw_dir,
        "json_dir": json_dir,
        "styles_csv": styles_csv,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
    }


@pytest.fixture
def patched_stage_modules(monkeypatch: pytest.MonkeyPatch, pipeline_paths: dict[str, Path]):
    """Redirect every stage module's path constants into the temp directory.

    Each stage script computes its INPUT_FILE / OUTPUT_FILE / etc. as
    module-level constants at import time, so patching src.config alone
    would not affect already-imported modules. We patch the constants on
    each already-imported module directly instead.
    """
    interim = pipeline_paths["interim_dir"]
    processed = pipeline_paths["processed_dir"]
    json_dir = pipeline_paths["json_dir"]
    styles_csv = pipeline_paths["styles_csv"]

    # Stage 02
    monkeypatch.setattr(stage02, "INPUT_FILE", styles_csv)
    monkeypatch.setattr(stage02, "JSON_DIRECTORY", json_dir)
    monkeypatch.setattr(stage02, "OUTPUT_FILE", interim / "garments_clean.csv")
    monkeypatch.setattr(stage02, "EXCLUDED_FILE", interim / "adult_scope_excluded.csv")
    monkeypatch.setattr(stage02, "CSV_REPAIR_AUDIT_FILE", interim / "styles_csv_parse_repairs.csv")

    # Stage 03b
    monkeypatch.setattr(stage03b, "INPUT_FILE", interim / "garments_clean.csv")
    monkeypatch.setattr(stage03b, "JSON_DIRECTORY", json_dir)
    monkeypatch.setattr(stage03b, "OUTPUT_FILE", interim / "garments_with_full_materials.csv")
    monkeypatch.setattr(stage03b, "AUDIT_FILE", interim / "material_audit_full_json.csv")

    # Stage 04
    monkeypatch.setattr(stage04, "INPUT_FILE", interim / "garments_with_full_materials.csv")
    monkeypatch.setattr(stage04, "OUTPUT_FILE", interim / "garments_with_material_scores.csv")
    monkeypatch.setattr(stage04, "AUDIT_FILE", interim / "material_score_audit.csv")

    # Stage 05
    monkeypatch.setattr(stage05, "INPUT_FILE", interim / "garments_with_material_scores.csv")
    monkeypatch.setattr(stage05, "OUTPUT_FILE", processed / "wardrobe_pilot.csv")
    monkeypatch.setattr(stage05, "EXCLUDED_FILE", processed / "wardrobe_excluded_no_material_score.csv")

    # Stage 06
    monkeypatch.setattr(stage06, "INPUT_FILE", processed / "wardrobe_pilot.csv")
    monkeypatch.setattr(stage06, "OUTPUT_FILE", processed / "wardrobe_scored.csv")

    return {
        "stage02": stage02, "stage03b": stage03b, "stage04": stage04,
        "stage05": stage05, "stage06": stage06,
    }



# Stage 02 cleaning / adult scoping


def test_stage02_keeps_only_adult_apparel(patched_stage_modules):
    stage02.clean_adult_apparel_metadata()

    clean_df = pd.read_csv(stage02.OUTPUT_FILE, dtype=str)
    kept_ids = set(clean_df["garment_id"])

    assert kept_ids == {"1001", "1002"}, (
        "Expected only the two adult Apparel garments (1001, 1002) to survive "
        f"stage 02. Got: {sorted(kept_ids)}"
    )

    excluded_df = pd.read_csv(stage02.EXCLUDED_FILE, dtype=str)
    excluded_ids = set(excluded_df["garment_id"])
    assert "1003" in excluded_ids, "Kids record (1003) should be excluded, not silently dropped."


def test_stage02_excludes_non_apparel_before_json_lookup(patched_stage_modules):
    """1004 (Footwear) has no JSON file at all this must not raise an
    error, because non-Apparel records are filtered out before any JSON
    lookup happens.
    """
    stage02.clean_adult_apparel_metadata()  # should not raise
    clean_df = pd.read_csv(stage02.OUTPUT_FILE, dtype=str)
    assert "1004" not in set(clean_df["garment_id"])



# Full chain: 02 -> 03b -> 04 -> 05 -> 06


def test_full_pipeline_chain_produces_valid_sustainability_scores(patched_stage_modules):
    """Run every stage in order and sanity-check the final output.

    This is the test most likely to catch a real cross-stage bug: if any
    stage renames, drops, or mistypes a column that a later stage expects,
    this test fails with a clear KeyError/AssertionError pointing at exactly
    which stage broke the contract.
    """
    stage02.clean_adult_apparel_metadata()
    stage03b.extract_full_json_materials()
    stage04.apply_material_scores()
    stage05.generate_wardrobe_data()
    stage06.calculate_final_scores()

    #  Stage 03b: material evidence extracted correctly 
    materials_df = pd.read_csv(stage03b.OUTPUT_FILE, dtype=str)
    materials_by_id = materials_df.set_index("garment_id")

    assert materials_by_id.loc["1001", "primary_material"] == "cotton"
    assert materials_by_id.loc["1001", "scoring_eligibility"] == "eligible_exact_single_fibre"

    assert materials_by_id.loc["1002", "primary_material"] == "cotton"
    assert materials_by_id.loc["1002", "secondary_materials"] == "polyester"
    assert materials_by_id.loc["1002", "scoring_eligibility"] == "known_blend_pending_policy"

    #  Stage 04: 1001 scored, 1002 left unscored (blend policy pending) 
    scores_df = pd.read_csv(stage04.OUTPUT_FILE, dtype={"garment_id": str})
    scores_by_id = scores_df.set_index("garment_id")

    assert scores_by_id.loc["1001", "material_score"] == pytest.approx(5.0)  # cotton benchmark
    assert pd.isna(scores_by_id.loc["1002", "material_score"])
    assert scores_by_id.loc["1002", "material_score_status"] == "unscored_blend_pending_policy"

    #  Stage 05: only scored garments enter the synthetic wardrobe 
    wardrobe_df = pd.read_csv(stage05.OUTPUT_FILE, dtype={"garment_id": str})
    assert set(wardrobe_df["garment_id"]) == {"1001"}, (
        "Only garments with a material score should reach the wardrobe stage."
    )
    # Wearability score exists and is within the documented 1.0-10.0 range.
    assert 1.0 <= wardrobe_df.loc[0, "wearability_score"] <= 10.0

    excluded_df = pd.read_csv(stage05.EXCLUDED_FILE, dtype={"garment_id": str})
    assert "1002" in set(excluded_df["garment_id"])

    #  Stage 06: final composite score is well-formed 
    final_df = pd.read_csv(stage06.OUTPUT_FILE, dtype={"garment_id": str})
    assert len(final_df) == 1
    row = final_df.iloc[0]

    assert 1.0 <= row["sustainability_score"] <= 10.0
    assert pd.isna(row["certification_score"]), (
        "certification_score must remain unassigned see Chapter 3, "
        "'Deviations from the Approved Proposal'."
    )
    assert row["certification_score_status"] == (
        "not_used_no_item_level_brand_or_certification_mapping"
    )
    assert row["sustainability_score_method"] == (
        "four_component_renormalised_score_with_multi_attribute_wearability"
    )



# Stage 03b isolated edge case: JSON file missing after cleaning


def test_stage03b_handles_missing_json_file_gracefully(
    monkeypatch: pytest.MonkeyPatch, pipeline_paths: dict[str, Path]
):
    """Simulate a garment_id present in garments_clean.csv whose JSON file
    has since gone missing (e.g. deleted between pipeline runs). This should
    be recorded as a missing-evidence row, not raise an exception.
    """
    interim = pipeline_paths["interim_dir"]
    json_dir = pipeline_paths["json_dir"]

    clean_csv = interim / "garments_clean.csv"
    pd.DataFrame(
        [{"garment_id": "9999", "gender": "Men", "master_category": "Apparel",
          "sub_category": "Topwear", "article_type": "Tshirts",
          "product_name": "Ghost Garment"}]
    ).to_csv(clean_csv, index=False)

    monkeypatch.setattr(stage03b, "INPUT_FILE", clean_csv)
    monkeypatch.setattr(stage03b, "JSON_DIRECTORY", json_dir)
    output_file = interim / "garments_with_full_materials_missing_test.csv"
    audit_file = interim / "material_audit_missing_test.csv"
    monkeypatch.setattr(stage03b, "OUTPUT_FILE", output_file)
    monkeypatch.setattr(stage03b, "AUDIT_FILE", audit_file)

    stage03b.extract_full_json_materials()  # should not raise

    result_df = pd.read_csv(output_file, dtype=str)
    row = result_df.iloc[0]
    assert row["json_file_found"] in ("False", "false")
    assert row["scoring_eligibility"] == "not_eligible_missing_material_evidence"
    assert row["exclusion_reason"] == "json_file_not_found"



# Stage 04 isolated edge case: recognised material with no benchmark entry


def test_stage04_leaves_unbenchmarked_material_unscored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A material the parser recognises but that has no entry in
    MATERIAL_LCI_BENCHMARKS must be left unscored, not silently defaulted.
    """
    input_file = tmp_path / "garments_with_full_materials.csv"
    pd.DataFrame(
        [{
            "garment_id": "5001",
            "primary_material": "cupro",  # not in MATERIAL_LCI_BENCHMARKS
            "secondary_materials": "",
            "scoring_eligibility": "eligible_exact_single_fibre",
            "exclusion_reason": "",
        }]
    ).to_csv(input_file, index=False)

    output_file = tmp_path / "garments_with_material_scores.csv"
    audit_file = tmp_path / "material_score_audit.csv"
    monkeypatch.setattr(stage04, "INPUT_FILE", input_file)
    monkeypatch.setattr(stage04, "OUTPUT_FILE", output_file)
    monkeypatch.setattr(stage04, "AUDIT_FILE", audit_file)

    stage04.apply_material_scores()

    result_df = pd.read_csv(output_file)
    row = result_df.iloc[0]
    assert pd.isna(row["material_score"])
    assert row["material_score_status"] == "unscored_missing_benchmark_policy"