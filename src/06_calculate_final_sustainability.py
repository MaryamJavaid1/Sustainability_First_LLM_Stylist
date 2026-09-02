"""06_calculate_final_sustainability.py

Calculates the final composite sustainability score for each synthetic
wardrobe garment.

Components and normalised weights (revised see Chapter 3, "Deviations from
the Approved Proposal"):

    material:      0.40 / 0.90 = 0.4444
    reuse:         0.25 / 0.90 = 0.2778
    wearability:   0.15 / 0.90 = 0.1667
    second_hand:   0.10 / 0.90 = 0.1111

A brand/certification component was in the original proposal's formula but
is not implemented here: the secondary dataset that would have supplied
item-level ESG/certification evidence was withdrawn, and no reliable
item-level mapping is otherwise available. The certification fields below
are kept as explicitly-flagged, unused audit columns rather than silently
dropped, so the omission is traceable in the output data itself.

Inputs:
    data/processed/wardrobe_pilot.csv

Outputs:
    data/processed/wardrobe_scored.csv
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import OUTPUTS, PROCESSED_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_pilot.csv"
OUTPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"

# Original project weights, excluding unavailable brand/certification data.
BASE_WEIGHTS = {
    "material": 0.40,
    "reuse": 0.25,
    "wearability": 0.15,
    "second_hand": 0.10,
}
WEIGHT_TOTAL = sum(BASE_WEIGHTS.values())
NORMALISED_WEIGHTS = {
    component: weight / WEIGHT_TOTAL for component, weight in BASE_WEIGHTS.items()
}


def convert_to_boolean(value: Any) -> bool:
    """Safely convert CSV values (various truthy string forms) to bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}


def calculate_final_scores() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 05_generate_wardrobe_attributes.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required_columns = {
        "material_score", "reuse_count", "wearability_score", "second_hand_status",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    df["material_score"] = pd.to_numeric(df["material_score"], errors="coerce")
    if df["material_score"].isna().any():
        missing_count = df["material_score"].isna().sum()
        raise ValueError(
            f"{missing_count:,} garments have no material score. "
            "Only scored garments should enter Step 06."
        )

    # Reuse count becomes a 0-10 score, capped at 100 wears.
    df["reuse_score"] = (
        df["reuse_count"].clip(lower=0, upper=100).div(100).mul(10).round(2)
    )

    df["wearability_score"] = pd.to_numeric(
        df["wearability_score"], errors="coerce"
    )
    if df["wearability_score"].isna().any():
        missing_count = df["wearability_score"].isna().sum()
        raise ValueError(
            f"{missing_count:,} garments have missing or invalid "
            "wearability scores. Only garments with a valid wearability "
            "score should enter Step 06."
        )
    df["condition_score"] = df["wearability_score"]  # backward-compatible alias

    df["second_hand_bonus"] = df["second_hand_status"].apply(
        lambda val: 10.0 if convert_to_boolean(val) else 0.0
    )

    # Certification/brand ESG is explicitly unassigned see module docstring
    # and Chapter 3, "Deviations from the Approved Proposal".
    df["certification_score"] = pd.NA
    df["certification_score_status"] = (
        "not_used_no_item_level_brand_or_certification_mapping"
    )

    raw_score = (
        df["material_score"] * NORMALISED_WEIGHTS["material"]
        + df["reuse_score"] * NORMALISED_WEIGHTS["reuse"]
        + df["wearability_score"] * NORMALISED_WEIGHTS["wearability"]
        + df["second_hand_bonus"] * NORMALISED_WEIGHTS["second_hand"]
    )
    df["sustainability_score"] = raw_score.clip(lower=1.0, upper=10.0).round(2)
    df["sustainability_score_method"] = (
        "four_component_renormalised_score_with_multi_attribute_wearability"
    )

    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Final sustainability scoring completed.")
    logger.info("Final dataset saved to: %s", OUTPUT_FILE)
    logger.info("Total scored garments: %s", f"{len(df):,}")
    logger.info("Normalised score weights used:")
    for component, weight in NORMALISED_WEIGHTS.items():
        logger.info("  %s: %.4f", component, weight)
    logger.info(
        "Sustainability score summary:\n%s",
        df["sustainability_score"].describe().round(2).to_string(),
    )


if __name__ == "__main__":
    calculate_final_scores()