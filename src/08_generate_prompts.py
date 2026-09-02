"""08_generate_prompts.py

Constructs the target JSONL dataset for later LLM fine-tuning. Simulates a
"Sustainability-First" AI stylist that prioritises garment reuse, LCA-backed
material impact, seasonal appropriateness, and occasion formality.

This script does NOT call or fine-tune an LLM. It produces rule-generated
prompt/response training pairs only fine-tuning is a separate pipeline
stage (the Colab notebook).

Selection logic mirrors app/recommender.py's rule_based_recommendation():
formality is filtered first, then season, with an explicit, disclosed
fallback at each stage rather than a silent one so the response text
never asserts a season match it didn't actually find.

Input:
    data/processed/wardrobe_scored.csv

Output:
    outputs/prompt_response_pairs.jsonl
"""

from __future__ import annotations

import json
import random
from typing import Any

import pandas as pd

from src.config import OUTPUTS, PROCESSED_DATA, RANDOM_SEED
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"
OUTPUT_FILE = OUTPUTS / "prompt_response_pairs.jsonl"

MIN_CLOSET_SIZE = 10
MAX_EVENTS_PER_OWNER = 3
MAX_WARDROBE_ITEMS_SHOWN = 15

# Maps contextual events to plausible seasons, to prevent mismatched outfit
# pairings (e.g. recommending a formal-dinner outfit built for a heatwave).
EVENT_SEASON_MAP = {
    "casual lunch": ["Summer", "Spring", "Fall"],
    "office meeting": ["Fall", "Winter", "Spring"],
    "university class": ["Fall", "Spring", "Winter"],
    "formal dinner": ["Winter", "Fall", "Summer"],
    "weekend errand": ["Summer", "Spring", "Fall", "Winter"],
}
DEFAULT_SEASONS = ["Summer", "Winter", "Fall", "Spring"]

# Maps each occasion to the garment usage/formality levels that suit it.
# Mirrors app/recommender.py's EVENT_FORMALITY_MAP, so training targets
# reflect the same occasion-appropriateness constraint the live rule-based
# engine actually applies. Previously absent from this script entirely
# meaning earlier training data could name formality-inappropriate items
# (e.g. innerwear for a formal dinner) with no constraint against it.
EVENT_FORMALITY_MAP: dict[str, list[str]] = {
    "casual lunch": ["casual", "smart casual", "sports"],
    "weekend errand": ["casual", "sports", "smart casual"],
    "university class": ["casual", "smart casual"],
    "office meeting": ["formal", "smart casual"],
    "formal dinner": ["formal", "smart casual", "ethnic"],
}


def _wardrobe_text(items: pd.DataFrame) -> str:
    """Render a wardrobe subset as a human-readable inventory list."""
    lines = [
        f"- {item['product_name']} [Category: {item['sub_category']}, "
        f"Season: {item['season']}] (Material: {item['extracted_material']}, "
        f"Worn: {item['reuse_count']} times, Score: {item['sustainability_score']})"
        for _, item in items.iterrows()
    ]
    return "\n".join(lines)


def _filter_to_formality(wardrobe: pd.DataFrame, event: str) -> pd.DataFrame:
    """Keep only garments whose usage/formality suits the occasion.

    Falls back to the unfiltered set if the filter would leave nothing, so
    a sparse wardrobe still produces a training example.
    """
    if "usage" not in wardrobe.columns:
        return wardrobe
    allowed = EVENT_FORMALITY_MAP.get(event)
    if not allowed:
        return wardrobe
    usage = wardrobe["usage"].fillna("").astype(str).str.strip().str.casefold()
    matches = wardrobe[usage.isin(allowed)]
    return matches if not matches.empty else wardrobe


def _select_outfit(season_items: pd.DataFrame) -> list[pd.Series]:
    """Select the highest-scoring top/dress/apparel-set and bottomwear for
    the outfit.

    A dress OR a coordinated apparel set (e.g. a kurta-with-dupatta set) is
    treated as a complete one-piece outfit on its own previously only
    "Dress" was recognised, meaning Apparel Set items were both excluded
    from the candidate pool entirely and never eligible to stand alone.
    """
    outfit_items: list[pd.Series] = []

    tops = season_items[
        season_items["sub_category"].str.contains("Topwear|Dress|Apparel Set", case=False, na=False)
    ]
    best_top = None
    if not tops.empty:
        best_top = tops.sort_values(by="sustainability_score", ascending=False).iloc[0]
        outfit_items.append(best_top)

    is_standalone = best_top is not None and (
        "Dress" in str(best_top.get("sub_category", ""))
        or "Apparel Set" in str(best_top.get("sub_category", ""))
    )
    if best_top is None or not is_standalone:
        bottoms = season_items[
            season_items["sub_category"].str.contains("Bottomwear", case=False, na=False)
        ]
        if not bottoms.empty:
            best_bottom = bottoms.sort_values(by="sustainability_score", ascending=False).iloc[0]
            outfit_items.append(best_bottom)

    if not outfit_items:
        outfit_items.append(
            season_items.sort_values(by="sustainability_score", ascending=False).iloc[0]
        )

    return outfit_items


def generate_sustainability_prompts() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 06_calculate_final_sustainability.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # Strictly filter out garments lacking sustainability scores, to avoid
    # generating training targets that reference unscored items.
    scored_df = df[df["sustainability_score"].notna()].copy()
    wardrobes = scored_df.groupby("wardrobe_owner_id")

    random.seed(RANDOM_SEED)
    prompts: list[dict[str, Any]] = []

    for _owner_id, closet in wardrobes:
        if len(closet) < MIN_CLOSET_SIZE:
            continue

        events = closet["event_context"].unique().tolist()

        for event in events[:MAX_EVENTS_PER_OWNER]:
            available_items = closet.sample(min(MAX_WARDROBE_ITEMS_SHOWN, len(closet)))
            possible_seasons = EVENT_SEASON_MAP.get(event, DEFAULT_SEASONS)
            current_season = random.choice(possible_seasons)

            wardrobe_text = _wardrobe_text(available_items)
            prompt = (
                f"User Request: I have a {event} today. It is currently "
                f"{current_season}. Please recommend a complete, sustainable "
                f"outfit from my wardrobe below that is appropriate for the "
                f"weather, rather than me buying something new.\n\n"
                f"My Wardrobe:\n{wardrobe_text}"
            )

            # Formality first previously absent entirely from this
            # script, meaning training targets could name formality-
            # inappropriate items with no constraint against it.
            formality_items = _filter_to_formality(available_items, event)

            # Then season, filtered against the SAME season stated in the
            # prompt. The fallback here is now explicit and disclosed in
            # the response text below, rather than silent previously,
            # a silent fallback let the response claim "perfectly suited
            # for {season} weather" even when no item actually matched
            # that season at all.
            season_items = formality_items[
                formality_items["season"].str.contains(current_season, case=False, na=False)
            ]
            season_matched = not season_items.empty
            if not season_matched:
                season_items = formality_items if not formality_items.empty else available_items

            outfit_items = _select_outfit(season_items)

            item_names = [
                f"your {item['product_name']} ({item['extracted_material']})"
                for item in outfit_items
            ]
            outfit_description = (
                " paired with ".join(item_names) if len(item_names) > 1 else item_names[0]
            )
            total_reuse = sum(item["reuse_count"] for item in outfit_items)

            if season_matched:
                season_clause = f"perfectly suited for the {current_season} weather"
            else:
                season_clause = (
                    "the best available option from your wardrobe, though not "
                    f"specifically tagged for {current_season}"
                )

            response = (
                f"I recommend wearing {outfit_description} for your {event}. "
                f"These items are {season_clause}. "
                f"More importantly, this combination prioritizes "
                f"lower-impact materials. Combined, these pieces have been "
                f"worn {total_reuse} times, which supports lower-impact reuse "
                f"compared to purchasing a new outfit. This is a practical, "
                f"sustainable choice!"
            )

            prompts.append({"instruction": prompt, "output": response})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item) + "\n")

    logger.info("Generated %s target sustainability prompt pairs.", len(prompts))
    logger.info("Saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    generate_sustainability_prompts()