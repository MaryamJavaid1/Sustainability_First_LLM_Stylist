"""06b_estimate_environmental_impact.py

Adds an estimated carbon footprint (kg CO2e) to each scored garment whose
primary material has a traceable per-kilogram carbon factor, delivering the
"estimated environmental impact" component described in the project
proposal (Objective 3).

This is an APPROXIMATION, and is documented as such. It multiplies two
factors:

    estimated_kg_co2e = material_co2_factor (kg CO2e per kg of fabric)
                        x estimated_garment_weight (kg, by article type)

Sources

Eight material CO2 factors (wool, acrylic, cotton, silk, nylon, polyester,
linen, viscose) are taken directly from WRAP (2012), "A Carbon Footprint
for UK Clothing and Opportunities for Savings" (Thomas, Fishwick, Joyce and
van Santen), Appendix IV, Table 21: "Carbon footprint of all clothing in
use in the UK in 2009 ... represented per tonne, broken down per fibre
type." The primary table was directly inspected (not inferred via
secondary corroboration) and gives a per-lifecycle-stage breakdown
fibre production, yarn production, fabric production, garment production,
distribution, retail, use (washing), use (drying), use (ironing), and end
of life (a negative credit) summing to a TOTAL column per tonne of
fibre. The factors below are that TOTAL column, converted from kg CO2e per
tonne to kg CO2e per kilogram by dividing by 1,000.

Because the TOTAL column includes in-use care impacts (washing, drying,
ironing) and an end-of-life credit, not production alone, this factor is
best understood as an indicative FULL-LIFECYCLE carbon intensity per
kilogram of fibre in use in the UK not a cradle-to-gate production-only
figure. This is a more complete, and arguably more meaningful,
counterfactual for "the impact of an equivalent new garment" than a
production-only figure would be, since a genuinely new garment would also
be laundered and eventually disposed of over its own life. This
distinction is stated explicitly in the dissertation (Section 4.11.2).

The remaining six materials in the dataset (bamboo, modal, leather, hemp,
lyocell, cashmere) have no traceable, unit-compatible published factor
identified during the per-factor provenance audit; these are set to None
and receive no environmental-impact estimate, rather than an invented
value.

Garment weights are indicative average dry weights by article type, drawn
from typical apparel weight ranges. They are approximations, not measured
values, and are documented as an assumption in the dissertation.

Interpretation

The figure represents an APPROXIMATE, INDICATIVE FULL-LIFECYCLE carbon
intensity estimate for an equivalent NEW garment of the same material and
estimated weight covering production, distribution/retail, in-use care,
and end-of-life, not a measured lifecycle figure for the specific physical
garment. Because ReWear recommends reusing an already-owned garment instead
of buying new, this value is presented in the application as a
COUNTERFACTUAL estimate: the approximate lifecycle impact that would be
associated with an equivalent new garment, under the assumption that
choosing reuse displaces a comparable new purchase one-for-one. This
assumption is not verified for any individual recommendation and is stated
as a limitation.

Inputs
    data/processed/wardrobe_scored.csv

Outputs

    data/processed/wardrobe_scored.csv   (same file, enriched in place)
"""

from __future__ import annotations

import pandas as pd

from src.config import OUTPUTS, PROCESSED_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"
OUTPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"

# Material carbon factors, kg CO2e per kg of fabric.
#
# Directly sourced from WRAP (2012), Appendix IV, Table 21, TOTAL column
# (kg CO2e per tonne of fibre), converted to a per-kilogram basis by
# dividing by 1,000. The primary table has been directly inspected this
# is no longer based on secondary corroboration alone.
MATERIAL_CO2_FACTORS_KG_PER_KG_VERIFIED: dict[str, float] = {
    "cotton": 27.68,
    "wool": 46.28,
    "silk": 25.43,
    "linen": 15.00,
    "viscose": 30.14,
    "polyester": 21.33,
    "acrylic": 38.43,
    "nylon": 24.35,
}

# Materials without a traceable, unit-compatible published source at the
# time of writing. Set to None rather than an invented value, consistent
# with the project's policy of not fabricating evidence (see Section
# 4.4-4.8 for the analogous policy applied to material composition). A
# garment whose primary material falls here receives no environmental-
# impact estimate.
UNVERIFIED_MATERIALS: dict[str, None] = {
    "bamboo": None,
    "modal": None,
    "leather": None,
    "hemp": None,
    "lyocell": None,
    "cashmere": None,
}

MATERIAL_CO2_FACTORS_KG_PER_KG: dict[str, float | None] = {
    **MATERIAL_CO2_FACTORS_KG_PER_KG_VERIFIED,
    **UNVERIFIED_MATERIALS,
}

# Indicative average dry garment weights in kg, by article type (lowercased).
# Approximations from typical apparel weight ranges; documented as an
# assumption, not a measured value. Unlisted article types use DEFAULT_WEIGHT.
GARMENT_WEIGHT_KG: dict[str, float] = {
    "tshirts": 0.18,
    "shirts": 0.22,
    "tops": 0.18,
    "tunics": 0.28,
    "kurtas": 0.30,
    "kurtis": 0.28,
    "sweaters": 0.45,
    "sweatshirts": 0.50,
    "jackets": 0.70,
    "blazers": 0.65,
    "dresses": 0.35,
    "sarees": 0.50,
    "jeans": 0.60,
    "trousers": 0.50,
    "track pants": 0.45,
    "shorts": 0.30,
    "leggings": 0.25,
    "skirts": 0.30,
    "capris": 0.35,
    "churidar": 0.30,
    "innerwear": 0.08,
    "briefs": 0.05,
    "bra": 0.08,
    "camisoles": 0.10,
}
DEFAULT_WEIGHT_KG = 0.30  # sensible mid-range fallback for unlisted types


def _garment_weight(article_type: str) -> float:
    key = str(article_type).strip().casefold()
    return GARMENT_WEIGHT_KG.get(key, DEFAULT_WEIGHT_KG)


def _material_factor(primary_material: str) -> float | None:
    key = str(primary_material).strip().casefold()
    return MATERIAL_CO2_FACTORS_KG_PER_KG.get(key)


def estimate_environmental_impact() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 06_calculate_final_sustainability.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = {"primary_material", "article_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    weights = df["article_type"].apply(_garment_weight)
    factors = df["primary_material"].apply(_material_factor)

    estimated = factors * weights
    df["estimated_garment_weight_kg"] = weights.round(3)
    df["material_co2_factor_kg_per_kg"] = factors
    df["estimated_production_co2e_kg"] = estimated.round(2)

    # Status flag so unscored/unknown-material rows are explicit, not silently 0.
    df["environmental_impact_status"] = "estimated_from_material_and_weight"
    df.loc[factors.isna(), "environmental_impact_status"] = (
        "not_estimated_no_material_co2_factor"
    )

    # Counterfactual estimate only: the approximate lifecycle footprint of
    # an equivalent NEW garment, under the assumption that reuse displaces a
    # comparable new purchase one-for-one. This is NOT a measured saving.
    df["counterfactual_new_garment_production_co2e_kg"] = df["estimated_production_co2e_kg"]

    df.to_csv(OUTPUT_FILE, index=False)

    estimable = df["estimated_production_co2e_kg"].notna()
    logger.info("Environmental impact estimation completed.")
    logger.info("Dataset enriched in place: %s", OUTPUT_FILE)
    logger.info("Garments with an impact estimate: %s", f"{estimable.sum():,}")
    logger.info("Garments without (no material factor): %s", f"{(~estimable).sum():,}")
    if estimable.any():
        logger.info(
            "Estimated production CO2e (kg) summary:\n%s",
            df.loc[estimable, "estimated_production_co2e_kg"].describe().round(2).to_string(),
        )


if __name__ == "__main__":
    estimate_environmental_impact()