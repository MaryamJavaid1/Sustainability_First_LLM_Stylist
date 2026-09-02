"""04_apply_lca_scores.py

Assigns evidence-informed Life Cycle Inventory (LCI) material impact proxy
scores to adult garments based on composition evidence.

Material benchmarks are the sole scoring source (Munasinghe, Druckman &
Dissanayake, 2021). No brand or certification data is used: the secondary
sustainability dataset that would have supported that component was
withdrawn (see Chapter 3, "Deviations from the Approved Proposal").

Inputs:
    data/interim/garments_with_full_materials.csv

Outputs:
    data/interim/garments_with_material_scores.csv
    data/interim/material_score_audit.csv
"""

from __future__ import annotations

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = INTERIM_DATA / "garments_with_full_materials.csv"
OUTPUT_FILE = INTERIM_DATA / "garments_with_material_scores.csv"
AUDIT_FILE = INTERIM_DATA / "material_score_audit.csv"

# Benchmark material proxy scores based on Munasinghe et al. (2021).
# Scale: 1.0 (highest impact / least sustainable) to 10.0 (lowest impact).
MATERIAL_LCI_BENCHMARKS: dict[str, float] = {
    "linen": 8.0,
    "bamboo": 7.5,
    "cotton": 5.0,
    "modal": 5.0,
    "wool": 4.5,
    "viscose": 4.0,
    "polyester": 4.0,
    "acrylic": 3.0,
    "nylon": 3.0,
    "leather": 2.0,
    "silk": 1.0,
}

SCORE_COLUMNS = [
    "material_score",
    "material_impact_proxy_score",
    "material_score_status",
    "material_score_basis",
]


def score_garment_material(row: pd.Series) -> pd.Series:
    """Assign a material_impact_proxy_score based on composition eligibility."""
    eligibility = str(row.get("scoring_eligibility", ""))
    primary_mat = str(row.get("primary_material", "")).strip().lower()

    if eligibility in {"eligible_exact_single_fibre", "main_material_proxy_only"}:
        score = MATERIAL_LCI_BENCHMARKS.get(primary_mat)
        status = (
            "scored_exact_single_fibre"
            if eligibility == "eligible_exact_single_fibre"
            else "scored_main_material_proxy"
        )

        if score is not None:
            basis = (
                f"LCI_benchmark_{primary_mat}"
                if eligibility == "eligible_exact_single_fibre"
                else f"LCI_proxy_{primary_mat}_unquantified_claim"
            )
            return pd.Series(
                {
                    "material_score": score,
                    "material_impact_proxy_score": score,
                    "material_score_status": status,
                    "material_score_basis": basis,
                }
            )

        return pd.Series(
            {
                "material_score": pd.NA,
                "material_impact_proxy_score": pd.NA,
                "material_score_status": "unscored_missing_benchmark_policy",
                "material_score_basis": f"no_LCI_benchmark_for_{primary_mat}",
            }
        )

    if eligibility == "known_blend_pending_policy":
        return pd.Series(
            {
                "material_score": pd.NA,
                "material_impact_proxy_score": pd.NA,
                "material_score_status": "unscored_blend_pending_policy",
                "material_score_basis": "blend_requires_documented_proxy_policy",
            }
        )

    if eligibility == "needs_role_review":
        return pd.Series(
            {
                "material_score": pd.NA,
                "material_impact_proxy_score": pd.NA,
                "material_score_status": "unscored_multi_part_composition",
                "material_score_basis": "multi_component_garment_requires_role_review",
            }
        )

    return pd.Series(
        {
            "material_score": pd.NA,
            "material_impact_proxy_score": pd.NA,
            "material_score_status": "unscored_insufficient_material_evidence",
            "material_score_basis": str(
                row.get("exclusion_reason", "insufficient_material_evidence")
            ),
        }
    )


def apply_material_scores() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 03b_extract_full_json_materials.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df.drop(columns=SCORE_COLUMNS, errors="ignore")

    scoring_results = df.apply(score_garment_material, axis=1)
    df = pd.concat(
        [df.reset_index(drop=True), scoring_results.reset_index(drop=True)], axis=1
    )

    df["material_score"] = pd.to_numeric(df["material_score"], errors="coerce")
    df["material_impact_proxy_score"] = pd.to_numeric(
        df["material_impact_proxy_score"], errors="coerce"
    )

    df.to_csv(OUTPUT_FILE, index=False)

    audit_mask = df["material_score"].isna() | df["material_score_status"].str.startswith(
        "unscored", na=False
    )
    audit_columns = [
        "garment_id", "product_name", "primary_material", "secondary_materials",
        "composition_status", "scoring_eligibility", "material_score",
        "material_score_status", "material_score_basis",
    ]
    audit_columns = [c for c in audit_columns if c in df.columns]
    df.loc[audit_mask, audit_columns].to_csv(AUDIT_FILE, index=False)

    logger.info("Material LCA proxy scoring completed.")
    logger.info("Scored dataset saved to: %s", OUTPUT_FILE)
    logger.info("Score audit file saved to: %s", AUDIT_FILE)
    logger.info("Total records: %s", f"{len(df):,}")
    logger.info(
        "Material-score status summary:\n%s",
        df["material_score_status"].value_counts(dropna=False).to_string(),
    )
    logger.info("Records with a material score: %s", f"{df['material_score'].notna().sum():,}")
    logger.info("Records without a material score: %s", f"{df['material_score'].isna().sum():,}")


if __name__ == "__main__":
    apply_material_scores()