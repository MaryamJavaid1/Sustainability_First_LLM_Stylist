"""04c_create_manual_review_sample.py

Creates a reproducible stratified manual-review sample from
material_manual_review_queue.csv.

The sample is for quality assurance and dissertation validation. It does not
replace the full audit dataset.
"""

from __future__ import annotations

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS, RANDOM_SEED
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = INTERIM_DATA / "material_manual_review_queue.csv"
OUTPUT_FILE = INTERIM_DATA / "material_manual_review_sample.csv"

# Sample sizes per review category. These are dissertation-methodology
# decisions (target ~300-record QA sample) do not change without updating
# the corresponding justification in the Methodology chapter.
SAMPLE_PLAN = {
    "needs_review": 212,
    "needs_percentage_review": 62,
    "needs_role_review": 26,
}


def create_manual_review_sample() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Manual review queue not found: {INPUT_FILE}\n"
            "Run 04b_create_material_audit.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required_columns = {"garment_id", "scoring_eligibility", "review_decision"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"The manual review queue is missing required columns: {sorted(missing_columns)}"
        )

    sampled_frames: list[pd.DataFrame] = []

    for index, (eligibility, requested_size) in enumerate(SAMPLE_PLAN.items()):
        group = df.loc[df["scoring_eligibility"].eq(eligibility)].copy()

        if group.empty:
            raise ValueError(f"No records found for: {eligibility}")

        sample_size = min(requested_size, len(group))
        sample = group.sample(n=sample_size, random_state=RANDOM_SEED + index).copy()
        sample["review_batch"] = eligibility
        sampled_frames.append(sample)

    sample_df = pd.concat(sampled_frames, ignore_index=True)

    sort_columns = [
        c for c in ["review_batch", "article_type", "product_name", "garment_id"]
        if c in sample_df.columns
    ]
    if sort_columns:
        sample_df = sample_df.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)

    sample_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Manual review sample created successfully.")
    logger.info("Output file: %s", OUTPUT_FILE)
    logger.info("Total sample rows: %s", f"{len(sample_df):,}")
    logger.info(
        "Sample breakdown:\n%s",
        sample_df["review_batch"].value_counts().to_string(),
    )


if __name__ == "__main__":
    create_manual_review_sample()