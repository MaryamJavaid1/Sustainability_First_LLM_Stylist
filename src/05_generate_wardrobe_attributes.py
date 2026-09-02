"""05_generate_wardrobe_attributes.py

Generates synthetic wardrobe attributes for garments that have a documented
material sustainability score, implementing a multi-attribute wearability
model based on physical wear indicators (Hall et al., 2025; Laitala & Klepp,
2020).

Only garments with an available material score enter the ready wardrobe
dataset. Records without a material score are retained separately in an
exclusion audit file rather than silently dropped.

Input:
    data/interim/garments_with_material_scores.csv

Outputs:
    data/processed/wardrobe_pilot.csv
    data/processed/wardrobe_excluded_no_material_score.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS, PROCESSED_DATA, RANDOM_SEED
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = INTERIM_DATA / "garments_with_material_scores.csv"
OUTPUT_FILE = PROCESSED_DATA / "wardrobe_pilot.csv"
EXCLUDED_FILE = PROCESSED_DATA / "wardrobe_excluded_no_material_score.csv"

# Synthetic-data generation parameters. These are dissertation-methodology
# assumptions (documented as such, not empirical measurements) do not
# change without updating the corresponding justification in the
# Methodology / Limitations chapter.
DAMAGE_PENALTIES = {
    "None": 0.0,
    "Minor": 2.0,
    "Moderate": 4.5,
    "Major": 7.5,
}
CARE_ADJUSTMENTS = {"High": 0.5, "Medium": 0.0, "Low": -1.0}
DAMAGE_OPTIONS = ["None", "Minor", "Moderate", "Major"]
DAMAGE_PROBS = [0.60, 0.25, 0.10, 0.05]
CARE_OPTIONS = ["High", "Medium", "Low"]
CARE_PROBS = [0.30, 0.50, 0.20]
N_WARDROBE_OWNERS = 200
EVENT_CONTEXTS = [
    "casual lunch",
    "office meeting",
    "university class",
    "formal dinner",
    "weekend errand",
]


def calculate_wearability_score(row: pd.Series) -> tuple[float, str]:
    """Calculate physical wearability score (1.0-10.0) and descriptive label.

    Supported by Hall et al. (2025) and Laitala & Klepp (2020), this model
    evaluates physical wearability using damage, care quality, age, and wear
    count, rather than relying solely on static wear thresholds.
    """
    damage = str(row.get("damage_level", "None"))
    care = str(row.get("care_quality", "Medium"))
    repaired = bool(row.get("repair_status", False))
    wears = int(row.get("reuse_count", 0))
    age_months = int(row.get("garment_age_months", 12))

    base_score = 10.0
    base_score -= DAMAGE_PENALTIES.get(damage, 1.0)
    base_score += CARE_ADJUSTMENTS.get(care, 0.0)

    if repaired and damage in {"Minor", "Moderate"}:
        base_score += 1.0

    wear_penalty = min(3.0, (wears / 50.0) * 1.5)
    age_penalty = min(1.5, (age_months / 36.0) * 1.0)
    base_score -= (wear_penalty + age_penalty)

    final_score = round(max(1.0, min(10.0, base_score)), 2)

    if final_score >= 8.0:
        label = "Excellent"
    elif final_score >= 6.0:
        label = "Good"
    elif final_score >= 4.0:
        label = "Fair"
    else:
        label = "Worn"

    return final_score, label



def _assign_gender_coherent_owners(
    df: pd.DataFrame,
    rng: "np.random.Generator",
    n_owners: int,
) -> "np.ndarray":
    """Assign each garment to a wardrobe owner of a compatible gender.

    Each of the ``n_owners`` synthetic owners is given a fixed gender
    (roughly half men, half women). A garment is then assigned only to an
    owner whose gender matches the garment's own gender; Unisex garments may
    go to any owner. This keeps each synthetic wardrobe internally coherent
    (a wardrobe never mixes men's and women's clothing), which matters
    because an outfit is later assembled from a single owner's wardrobe.

    Falls back to a purely random owner when a garment has no usable gender.
    """
    # Split the owner pool into men's and women's owners.
    owner_ids = np.arange(1, n_owners + 1)
    is_men_owner = rng.random(n_owners) < 0.5
    men_owners = owner_ids[is_men_owner]
    women_owners = owner_ids[~is_men_owner]

    # Guard against an unlucky all-one-gender split.
    if men_owners.size == 0:
        men_owners = owner_ids[:1]
    if women_owners.size == 0:
        women_owners = owner_ids[-1:]

    genders = df["gender"].fillna("").astype(str).str.strip().str.casefold().to_numpy()
    assigned = np.empty(len(df), dtype=int)

    for i, gender in enumerate(genders):
        if gender == "men":
            pool = men_owners
        elif gender == "women":
            pool = women_owners
        else:  # unisex or unknown -> any owner
            pool = owner_ids
        assigned[i] = pool[rng.integers(0, pool.size)]

    return assigned


def _ensure_top_and_bottom_per_owner(
    df: pd.DataFrame,
    rng: "np.random.Generator",
) -> pd.DataFrame:
    """Guarantee every wardrobe has at least one topwear and one bottomwear.

    Synthetic assignment is random, so some owners can end up with only tops
    or only bottoms, which prevents a complete outfit from being built. This
    pass gives each such owner one extra garment of the missing type, taken
    from an owner of the same gender who has a surplus (more than one) of that
    type. Formality distribution is deliberately NOT balanced here, so
    wardrobes remain realistically uneven (not everyone owns formalwear).

    This changes the owner assignment of a small number of garments only;
    all other synthetic attributes are unaffected.
    """
    sub = df["sub_category"].fillna("").astype(str).str.casefold()
    is_top = sub.str.contains("topwear|dress", na=False)
    is_bottom = sub.str.contains("bottomwear", na=False)
    df = df.copy()
    df["_is_top"] = is_top.to_numpy()
    df["_is_bottom"] = is_bottom.to_numpy()
    df["_gender_key"] = df["gender"].fillna("").astype(str).str.strip().str.casefold()

    for kind, flag in (("_is_top", "top"), ("_is_bottom", "bottom")):
        counts = df.groupby("wardrobe_owner_id")[kind].sum()
        owners_missing = counts[counts == 0].index.tolist()

        for owner in owners_missing:
            owner_gender = (
                df.loc[df["wardrobe_owner_id"] == owner, "_gender_key"].mode()
            )
            owner_gender = owner_gender.iloc[0] if not owner_gender.empty else ""

            # Candidate donors: garments of the needed type, from owners who
            # have more than one of that type, gender-compatible.
            donor_counts = df.groupby("wardrobe_owner_id")[kind].transform("sum")
            gender_ok = (df["_gender_key"] == owner_gender) | (
                df["_gender_key"] == "unisex"
            )
            donors = df[
                df[kind]
                & (donor_counts > 1)
                & (df["wardrobe_owner_id"] != owner)
                & gender_ok
            ]
            if donors.empty:
                donors = df[df[kind] & (donor_counts > 1) & (df["wardrobe_owner_id"] != owner)]
            if donors.empty:
                continue  # no surplus anywhere; leave as-is (rare)

            pick = donors.sample(n=1, random_state=int(rng.integers(0, 1_000_000)))
            df.loc[pick.index, "wardrobe_owner_id"] = owner

    return df.drop(columns=["_is_top", "_is_bottom", "_gender_key"])


def generate_wardrobe_data() -> None:
    """Create reproducible synthetic wardrobe ownership and use attributes."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\nRun 04_apply_lca_scores.py first."
        )

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if "material_score" not in df.columns:
        raise KeyError("The input dataset does not contain the 'material_score' column.")

    total_records = len(df)

    excluded_df = df[df["material_score"].isna()].copy()
    excluded_df.to_csv(EXCLUDED_FILE, index=False)

    df = df[df["material_score"].notna()].copy()
    if df.empty:
        raise ValueError(
            "No garments have a material score. Check the output of 04_apply_lca_scores.py."
        )

    if "gender" not in df.columns:
        raise KeyError(
            "The input dataset does not contain the 'gender' column, "
            "which is required for gender-coherent wardrobe assignment."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)

    df["wardrobe_owner_id"] = _assign_gender_coherent_owners(
        df, rng, N_WARDROBE_OWNERS
    )
    # Guarantee each wardrobe can form a complete outfit (>=1 top, >=1 bottom).
    df = _ensure_top_and_bottom_per_owner(df, rng)
    df["garment_age_months"] = rng.integers(low=1, high=61, size=n)

    monthly_wear_rate = rng.uniform(low=0.5, high=3.0, size=n)
    df["reuse_count"] = np.floor(df["garment_age_months"] * monthly_wear_rate).astype(int)

    df["last_worn_days"] = rng.integers(low=0, high=365, size=n)

    df["damage_level"] = rng.choice(DAMAGE_OPTIONS, size=n, p=DAMAGE_PROBS)
    df["care_quality"] = rng.choice(CARE_OPTIONS, size=n, p=CARE_PROBS)
    df["repair_status"] = rng.choice([True, False], size=n, p=[0.08, 0.92])
    df["second_hand_status"] = rng.choice([True, False], size=n, p=[0.15, 0.85])

    wearability_scores: list[float] = []
    wearability_labels: list[str] = []
    for _, row in df.iterrows():
        score, label = calculate_wearability_score(row)
        wearability_scores.append(score)
        wearability_labels.append(label)

    df["wearability_score"] = wearability_scores
    df["wearability_label"] = wearability_labels
    df["condition"] = wearability_labels  # backward-compatible alias

    df["event_context"] = rng.choice(EVENT_CONTEXTS, size=n)

    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Synthetic wardrobe generation completed.")
    logger.info("Ready wardrobe dataset saved to: %s", OUTPUT_FILE)
    logger.info("Excluded-record audit saved to: %s", EXCLUDED_FILE)
    logger.info("Total input garments: %s", f"{total_records:,}")
    logger.info("Ready garments with material scores: %s", f"{len(df):,}")
    logger.info("Excluded garments without material scores: %s", f"{len(excluded_df):,}")
    logger.info(
        "Wearability label summary:\n%s",
        df["wearability_label"].value_counts().to_string(),
    )
    logger.info(
        "Damage level summary:\n%s",
        df["damage_level"].value_counts().to_string(),
    )
    logger.info(
        "Second-hand status summary:\n%s",
        df["second_hand_status"].value_counts().to_string(),
    )


if __name__ == "__main__":
    generate_wardrobe_data()