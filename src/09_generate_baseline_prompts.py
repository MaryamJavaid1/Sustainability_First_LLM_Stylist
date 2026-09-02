"""09_generate_baseline_prompts.py

Constructs a rule-based, style-first comparison baseline. This model
prioritises item "newness" (lowest reuse count), actively ignoring LCA
scores and sustainability data, so that it can be compared against the
sustainability-first model during evaluation.

This is a rule-based experimental baseline constructed for this study it
does not represent how any real commercial recommendation engine works, and
it still recommends items from the existing wardrobe rather than suggesting
new purchases. Its "fast-fashion" behaviour is simulated only by
prioritising less-worn items.

Formality and season handling deliberately mirror
08_generate_prompts.py's, so the two datasets differ ONLY in their ranking
criterion (sustainability score vs. reuse count) and in whether
sustainability data is shown to the wardrobe not in unrelated
context-awareness capabilities. This keeps the sustainability-first vs.
style-first comparison a controlled one.

Input:
    data/processed/wardrobe_scored.csv

Output:
    outputs/baseline_prompt_response_pairs.jsonl
"""

from __future__ import annotations

import json
import random
from typing import Any

import pandas as pd

from src.config import OUTPUTS, PROCESSED_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"
OUTPUT_FILE = OUTPUTS / "baseline_prompt_response_pairs.jsonl"

MIN_CLOSET_SIZE = 10
MAX_EVENTS_PER_OWNER = 3
MAX_WARDROBE_ITEMS_SHOWN = 15

# Distinct seed from 08_generate_prompts.py so baseline pairings differ from
# the sustainability-first dataset.
BASELINE_RANDOM_SEED = 999

# Mirrors 08_generate_prompts.py's EVENT_SEASON_MAP. Previously absent from
# this script entirely the baseline had zero season-awareness, which
# would have let it recommend, for example, a heavy wool sweater for a
# summer casual lunch. Adding this keeps the comparison against the
# sustainability-first dataset controlled: both now share the same
# context-awareness, differing only in ranking criterion.
EVENT_SEASON_MAP = {
    "casual lunch": ["Summer", "Spring", "Fall"],
    "office meeting": ["Fall", "Winter", "Spring"],
    "university class": ["Fall", "Spring", "Winter"],
    "formal dinner": ["Winter", "Fall", "Summer"],
    "weekend errand": ["Summer", "Spring", "Fall", "Winter"],
}
DEFAULT_SEASONS = ["Summer", "Winter", "Fall", "Spring"]

# Mirrors 08_generate_prompts.py's EVENT_FORMALITY_MAP. Previously absent
# from this script entirely.
EVENT_FORMALITY_MAP: dict[str, list[str]] = {
    "casual lunch": ["casual", "smart casual", "sports"],
    "weekend errand": ["casual", "sports", "smart casual"],
    "university class": ["casual", "smart casual"],
    "office meeting": ["formal", "smart casual"],
    "formal dinner": ["formal", "smart casual", "ethnic"],
}


def _baseline_wardrobe_text(items: pd.DataFrame) -> str:
    """Render a wardrobe subset without any sustainability-related fields
    (wear count, condition, LCA score are deliberately omitted see
    module docstring). Season is included, since it is basic contextual
    information, not sustainability data."""
    lines = [
        f"- {item['product_name']} [Category: {item['sub_category']}, "
        f"Season: {item['season']}, Color: {item['base_colour']}]"
        for _, item in items.iterrows()
    ]
    return "\n".join(lines)


def _filter_to_formality(wardrobe: pd.DataFrame, event: str) -> pd.DataFrame:
    """Keep only garments whose usage/formality suits the occasion.
    Falls back to the unfiltered set if the filter would leave nothing."""
    if "usage" not in wardrobe.columns:
        return wardrobe
    allowed = EVENT_FORMALITY_MAP.get(event)
    if not allowed:
        return wardrobe
    usage = wardrobe["usage"].fillna("").astype(str).str.strip().str.casefold()
    matches = wardrobe[usage.isin(allowed)]
    return matches if not matches.empty else wardrobe


def _select_baseline_outfit(available_items: pd.DataFrame) -> list[pd.Series]:
    """Select outfit items using style-first logic (lowest reuse = 'newest').

    A dress or coordinated apparel set is treated as a complete one-piece
    outfit on its own previously only "Dress" was recognised, meaning
    Apparel Set items were both excluded from the candidate pool entirely
    and never eligible to stand alone.
    """
    outfit_items: list[pd.Series] = []

    tops = available_items[
        available_items["sub_category"].str.contains("Topwear|Dress|Apparel Set", case=False, na=False)
    ]
    best_top = None
    if not tops.empty:
        best_top = tops.sort_values(by="reuse_count", ascending=True).iloc[0]
        outfit_items.append(best_top)

    is_standalone = best_top is not None and (
        "Dress" in str(best_top.get("sub_category", ""))
        or "Apparel Set" in str(best_top.get("sub_category", ""))
    )
    if best_top is None or not is_standalone:
        bottoms = available_items[
            available_items["sub_category"].str.contains("Bottomwear", case=False, na=False)
        ]
        if not bottoms.empty:
            outfit_items.append(bottoms.sample(1).iloc[0])

    if not outfit_items:
        outfit_items.append(available_items.sample(1).iloc[0])

    return outfit_items


def generate_baseline_prompts() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 06_calculate_final_sustainability.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    scored_df = df[df["sustainability_score"].notna()].copy()
    wardrobes = scored_df.groupby("wardrobe_owner_id")

    random.seed(BASELINE_RANDOM_SEED)
    prompts: list[dict[str, Any]] = []

    for _owner_id, closet in wardrobes:
        if len(closet) < MIN_CLOSET_SIZE:
            continue

        events = closet["event_context"].unique().tolist()

        for event in events[:MAX_EVENTS_PER_OWNER]:
            available_items = closet.sample(min(MAX_WARDROBE_ITEMS_SHOWN, len(closet)))
            possible_seasons = EVENT_SEASON_MAP.get(event, DEFAULT_SEASONS)
            current_season = random.choice(possible_seasons)

            # The baseline prompt hides sustainability data (wear count,
            # condition, LCA score) this is deliberate, see module
            # docstring. Season and formality context are now included,
            # matching 08_generate_prompts.py, to keep the comparison
            # controlled.
            wardrobe_text = _baseline_wardrobe_text(available_items)
            prompt = (
                f"User Request: I have a {event} today. It is currently "
                f"{current_season}. Please recommend a stylish outfit from "
                f"my wardrobe below that is appropriate for the weather.\n\n"
                f"My Wardrobe:\n{wardrobe_text}"
            )

            formality_items = _filter_to_formality(available_items, event)

            season_items = formality_items[
                formality_items["season"].str.contains(current_season, case=False, na=False)
            ]
            season_matched = not season_items.empty
            if not season_matched:
                season_items = formality_items if not formality_items.empty else available_items

            outfit_items = _select_baseline_outfit(season_items)

            item_names = [f"your {item['product_name']}" for item in outfit_items]
            outfit_description = (
                " paired with ".join(item_names) if len(item_names) > 1 else item_names[0]
            )

            if season_matched:
                season_clause = f", well suited for the {current_season} weather"
            else:
                season_clause = ""

            # Previously this always asserted "the colours coordinate well"
            # regardless of the bottom item being selected at random, with
            # no colour-matching logic anywhere in the code an
            # unverified claim baked into every training target. The
            # justification now grounds itself only in what was actually
            # computed: reuse count, matching the same "freshest,
            # least-worn" language already used for style-first mode in
            # app/recommender.py.
            response = (
                f"For your {event}, I recommend wearing {outfit_description}"
                f"{season_clause}. These are among the freshest, least-worn "
                f"pieces in your wardrobe, giving your look a current, "
                f"put-together feel for the occasion."
            )

            prompts.append({"instruction": prompt, "output": response})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item) + "\n")

    logger.info(
        "Generated %s evaluation baseline (style-first) prompt pairs.", len(prompts)
    )
    logger.info("Saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    generate_baseline_prompts()