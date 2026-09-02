"""04d_score_blends.py

Scores garments classified as `known_blend_pending_policy` by stage 04, using
a documented, evidence-based blend policy: the renormalised, composition-
weighted average of the LCA benchmark scores of the blend's components.

Policy:

For a blend with components (material_i, percentage_i):

    blend_score = sum_over_benchmarked(percentage_i * benchmark_i)
                  / sum_over_benchmarked(percentage_i)

Only components that HAVE an LCA benchmark contribute; the weights are
renormalised over those components (see meeting decision). This means a blend
such as "95% cotton, 5% elastane" is scored on the cotton fraction alone,
because elastane has no benchmark. Blends where NO component is benchmarked
remain unscored and are reported.

This is a transparent, non-arbitrary rule consistent with how blended-textile
LCA impacts are commonly approximated, and it uses the exact composition
percentages already extracted by the material-evidence parser. It is applied
only to garments the parser judged to have reliable, quantified blend
composition (`known_blend_pending_policy`); it does not affect any other
category.

Inputs
    data/interim/garments_with_material_scores.csv

Outputs

    data/interim/garments_with_material_scores.csv   (enriched in place)
    data/interim/blend_scoring_audit.csv
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS
from src.logging_utils import get_logger

# Reuse the SAME benchmark table as stage 04, so single-fibre and blend
# scores are on one consistent scale.
from importlib import import_module

_stage04 = import_module("src.04_apply_lca_scores")
MATERIAL_LCI_BENCHMARKS: dict[str, float] = _stage04.MATERIAL_LCI_BENCHMARKS

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = INTERIM_DATA / "garments_with_material_scores.csv"
OUTPUT_FILE = INTERIM_DATA / "garments_with_material_scores.csv"
AUDIT_FILE = INTERIM_DATA / "blend_scoring_audit.csv"

TARGET_ELIGIBILITY = "known_blend_pending_policy"


def _parse_components(raw_json: Any) -> list[dict[str, Any]]:
    """Parse composition_percentages_json into a list of {material, percentage}."""
    if not isinstance(raw_json, str) or not raw_json.strip():
        return []
    try:
        parsed = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []

    components: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        material = (
            entry.get("canonical_material")
            or entry.get("material")
            or ""
        )
        percentage = entry.get("percentage")
        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            continue
        material = str(material).strip().casefold()
        if material and percentage > 0:
            components.append({"material": material, "percentage": percentage})
    return components


def _blend_score(components: list[dict[str, Any]]) -> tuple[float | None, str, str]:
    """Return (score, status, basis) for a blend using renormalised weights."""
    benchmarked = [
        c for c in components if c["material"] in MATERIAL_LCI_BENCHMARKS
    ]
    if not benchmarked:
        return None, "unscored_blend_no_benchmarked_component", "no_component_has_benchmark"

    weight_total = sum(c["percentage"] for c in benchmarked)
    if weight_total <= 0:
        return None, "unscored_blend_zero_weight", "benchmarked_weight_sum_zero"

    weighted = sum(
        c["percentage"] * MATERIAL_LCI_BENCHMARKS[c["material"]] for c in benchmarked
    )
    score = round(weighted / weight_total, 2)

    covered = round(weight_total, 1)
    parts = "+".join(
        f"{c['material']}:{c['percentage']:.0f}%" for c in benchmarked
    )
    basis = f"renormalised_weighted_avg[{parts}]_covered={covered}%"
    status = (
        "scored_blend_weighted_average"
        if weight_total >= 99.5
        else "scored_blend_weighted_average_partial_coverage"
    )
    return score, status, basis


def score_blends() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\nRun 04_apply_lca_scores.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    for col in ("scoring_eligibility", "composition_percentages_json"):
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    blend_mask = df["scoring_eligibility"].eq(TARGET_ELIGIBILITY)
    n_blends = int(blend_mask.sum())

    scored_count = 0
    unscored_count = 0
    audit_rows: list[dict[str, Any]] = []

    for idx in df.index[blend_mask]:
        components = _parse_components(df.at[idx, "composition_percentages_json"])
        score, status, basis = _blend_score(components)

        if score is not None:
            df.at[idx, "material_score"] = score
            df.at[idx, "material_impact_proxy_score"] = score
            df.at[idx, "material_score_status"] = status
            df.at[idx, "material_score_basis"] = basis
            scored_count += 1
        else:
            df.at[idx, "material_score_status"] = status
            df.at[idx, "material_score_basis"] = basis
            unscored_count += 1

        audit_rows.append(
            {
                "garment_id": df.at[idx, "garment_id"] if "garment_id" in df.columns else "",
                "product_name": df.at[idx, "product_name"] if "product_name" in df.columns else "",
                "composition_percentages_json": df.at[idx, "composition_percentages_json"],
                "assigned_material_score": score,
                "material_score_status": status,
                "material_score_basis": basis,
            }
        )

    df["material_score"] = pd.to_numeric(df["material_score"], errors="coerce")
    df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False)

    logger.info("Blend scoring completed.")
    logger.info("Blends targeted (known_blend_pending_policy): %s", f"{n_blends:,}")
    logger.info("Blends newly scored: %s", f"{scored_count:,}")
    logger.info("Blends still unscored (no benchmarked component): %s", f"{unscored_count:,}")
    logger.info("Enriched dataset saved to: %s", OUTPUT_FILE)
    logger.info("Blend audit saved to: %s", AUDIT_FILE)
    total_scored = int(df["material_score"].notna().sum())
    logger.info("Total garments with a material score now: %s", f"{total_scored:,}")


if __name__ == "__main__":
    score_blends()