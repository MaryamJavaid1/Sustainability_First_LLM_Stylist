"""04b_create_material_audit.py

Creates two audit outputs from the latest material-scoring dataset:

1. material_audit.csv
   Complete audit of every garment without a material score.

2. material_manual_review_queue.csv
   Smaller queue containing only records that need a human decision about
   material evidence, percentages, or multiple garment parts.

This prevents policy-only cases, such as known blends awaiting a documented
LCA policy, from being confused with records requiring manual data review.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = INTERIM_DATA / "garments_with_material_scores.csv"
AUDIT_FILE = INTERIM_DATA / "material_audit.csv"
MANUAL_REVIEW_FILE = INTERIM_DATA / "material_manual_review_queue.csv"

MANUAL_REVIEW_ELIGIBILITIES = {
    "needs_review",
    "needs_percentage_review",
    "needs_role_review",
}


def has_recorded_review_decisions(path: Path) -> bool:
    """Return True only when an existing manual-review file contains
    reviewer decisions. Prevents accidental loss of completed QA work.
    """
    if not path.exists():
        return False

    try:
        existing = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return False

    if "review_decision" not in existing.columns:
        return False

    decisions = existing["review_decision"].fillna("").str.strip()
    return bool(decisions.ne("").any())


def ensure_safe_to_regenerate() -> None:
    """Do not overwrite a manual-review queue that already contains human
    decisions.
    """
    protected_files = [
        path for path in [MANUAL_REVIEW_FILE] if has_recorded_review_decisions(path)
    ]

    if protected_files:
        files_text = "\n".join(str(path) for path in protected_files)
        raise FileExistsError(
            "Manual review decisions already exist in:\n"
            f"{files_text}\n\n"
            "Rename or back up that file before generating a new queue."
        )


def add_readable_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Create readable aliases when upstream data uses CSV-prefixed names."""
    df = df.copy()
    aliases = {
        "product_name": "csv_product_name",
        "gender": "csv_gender",
        "master_category": "csv_master_category",
        "sub_category": "csv_sub_category",
        "article_type": "csv_article_type",
    }
    for preferred_name, fallback_name in aliases.items():
        if preferred_name not in df.columns and fallback_name in df.columns:
            df[preferred_name] = df[fallback_name]
    return df


def classify_audit_action(row: pd.Series) -> str:
    """Assign a clear next action for every unscored material record."""
    eligibility = str(row.get("scoring_eligibility", ""))
    score_status = str(row.get("material_score_status", ""))

    if eligibility in MANUAL_REVIEW_ELIGIBILITIES:
        return "manual_evidence_review"
    if score_status == "unscored_multi_part_composition":
        return "manual_evidence_review"
    if score_status == "unscored_blend_pending_policy":
        return "blend_policy_required"
    if score_status == "unscored_missing_benchmark_policy":
        return "benchmark_policy_required"
    if score_status == "unscored_insufficient_material_evidence":
        return "no_reliable_material_evidence"
    return "review_pipeline_status"


def select_available_columns(df: pd.DataFrame, requested_columns: list[str]) -> pd.DataFrame:
    """Keep only columns that exist in the current pipeline output."""
    available_columns = [c for c in requested_columns if c in df.columns]
    return df[available_columns].copy()


def create_material_audit() -> None:
    """Create complete and manual-review audit files from unscored records."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\nRun 04_apply_lca_scores.py first."
        )

    INTERIM_DATA.mkdir(parents=True, exist_ok=True)
    ensure_safe_to_regenerate()

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = add_readable_column_aliases(df)

    required_columns = {"material_score", "material_score_status", "scoring_eligibility"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"The material-score dataset is missing required columns: {sorted(missing_columns)}"
        )

    df["material_score"] = pd.to_numeric(df["material_score"], errors="coerce")
    score_status = df["material_score_status"].astype("string")

    unscored_mask = df["material_score"].isna() | score_status.str.startswith(
        "unscored_", na=False
    )
    audit_df = df.loc[unscored_mask].copy()
    audit_df["audit_action"] = audit_df.apply(classify_audit_action, axis=1)
    audit_df["audit_reason"] = audit_df["material_score_basis"].fillna("")

    audit_columns = [
        "garment_id", "gender", "master_category", "sub_category", "article_type",
        "base_colour", "season", "usage", "product_name", "json_product_display_name",
        "primary_material", "secondary_materials", "material_variants",
        "extracted_material", "composition_status", "percentage_total",
        "percentage_total_status", "material_role_flags", "material_source",
        "material_confidence", "selected_material_evidence_source",
        "selected_material_evidence_excerpt", "scoring_eligibility",
        "material_score", "material_score_status", "material_score_basis",
        "audit_action", "audit_reason",
    ]
    audit_df = select_available_columns(audit_df, audit_columns)

    sort_columns = [
        c for c in ["audit_action", "scoring_eligibility", "primary_material",
                     "article_type", "product_name"]
        if c in audit_df.columns
    ]
    if sort_columns:
        audit_df = audit_df.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)

    manual_review_df = audit_df.loc[
        audit_df["audit_action"].eq("manual_evidence_review")
    ].copy()
    manual_review_df["review_decision"] = ""
    manual_review_df["verified_primary_material"] = ""
    manual_review_df["verified_secondary_materials"] = ""
    manual_review_df["verified_material_evidence"] = ""
    manual_review_df["review_notes"] = ""

    audit_df.to_csv(AUDIT_FILE, index=False)
    manual_review_df.to_csv(MANUAL_REVIEW_FILE, index=False)

    logger.info("Material audit created successfully.")
    logger.info("Complete audit file: %s", AUDIT_FILE)
    logger.info("Manual review queue: %s", MANUAL_REVIEW_FILE)
    logger.info("Total unscored records: %s", f"{len(audit_df):,}")
    logger.info("Manual evidence-review records: %s", f"{len(manual_review_df):,}")
    logger.info(
        "Breakdown by audit action:\n%s",
        audit_df["audit_action"].value_counts().to_string(),
    )
    logger.info(
        "Breakdown by material-score status:\n%s",
        audit_df["material_score_status"].value_counts(dropna=False).to_string(),
    )


if __name__ == "__main__":
    create_material_audit()