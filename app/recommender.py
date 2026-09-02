"""recommender.py

Core recommendation logic for the Sustainability-First Stylist web app.

This module deliberately reuses the SAME rule-based, season-aware, highest-
sustainability-score outfit-selection logic as src/08_generate_prompts.py, so
the web app demonstrates the actual pipeline behaviour rather than a separate
ad-hoc heuristic.

Two recommendation paths are provided:

    1. rule_based_recommendation() no LLM, no API key, always available.
       Used as the automatic fallback so the app never fails in a demo.

    2. llm_recommendation() optional live Google Gemini call.
       Used only when a GEMINI_API_KEY is configured AND the user asked for it.
       The base model is prompted with sustainability-first (or style-first)
       instructions. NOTE: this is prompt-engineered guidance on a base model,
       NOT the fine-tuned model described in the proposal the dissertation
       must state this distinction plainly.

Both paths, and the specific-garment-type path within them, ultimately return
an OutfitResult. Callers should treat OutfitResult.fallback_note as a
user-facing disclosure: whenever a constraint (season, formality, requested
garment type) had to be relaxed to produce a recommendation, that relaxation
is reported explicitly rather than silently substituted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "rewear-sustainability-q8"

# Configuration

# Ordered season preferences per occasion, mirroring 08_generate_prompts.py.
# Order matters: _filter_to_season_preference() tries these left-to-right and
# stops at the first season with any matching stock, so earlier entries are
# effectively "preferred" seasons for that occasion.
EVENT_SEASON_MAP: dict[str, list[str]] = {
    "casual lunch": ["Summer", "Spring", "Fall"],
    "office meeting": ["Fall", "Winter", "Spring"],
    "university class": ["Fall", "Spring", "Winter"],
    "formal dinner": ["Winter", "Fall", "Summer"],
    "weekend errand": ["Summer", "Spring", "Fall", "Winter"],
}

# Used when an event has no explicit entry in EVENT_SEASON_MAP.
DEFAULT_SEASONS: list[str] = ["Summer", "Winter", "Fall", "Spring"]

# Maps each occasion to the garment usage/formality levels that suit it. The
# dataset's `usage` field takes values such as Casual, Formal, Smart Casual,
# Ethnic, Sports. Filtering by these before ranking means the occasion
# genuinely shapes the recommendation (e.g. a formal dinner will not return a
# casual graphic t-shirt). Values are lowercased for matching.
EVENT_FORMALITY_MAP: dict[str, list[str]] = {
    "casual lunch": ["casual", "smart casual", "sports"],
    "weekend errand": ["casual", "sports", "smart casual"],
    "university class": ["casual", "smart casual"],
    "office meeting": ["formal", "smart casual"],
    "formal dinner": ["formal", "smart casual", "ethnic"],
}

# Article types whose dataset label is grammatically plural (e.g. "Sarees",
# "Jackets") but should read as singular in prose when exactly one item is
# being recommended (e.g. "your highest-scoring saree", not "...sarees").
ARTICLE_TYPE_SINGULAR: dict[str, str] = {
    "Sarees": "saree",
    "Jeans": "jeans",  # already grammatically singular-usable
    "Trousers": "trousers",
    "Jeggings": "jeggings",
    "Kurtas": "kurta",
    "Kurtis": "kurti",
    "Kurta Sets": "kurta set",
    "Tshirts": "t-shirt",
    "Tops": "top",
    "Tunics": "tunic",
    "Dresses": "dress",
    "Shorts": "shorts",
    "Lounge Shorts": "lounge shorts",
    "Skirts": "skirt",
    "Leggings": "leggings",
    "Jackets": "jacket",
    "Sweaters": "sweater",
    "Sweatshirts": "sweatshirt",
    "Tracksuits": "tracksuit",
    "Shirts": "shirt",
    "Track Pants": "track pants",
    "Lounge Pants": "lounge pants",
    "Capris": "capris",
}

# Maps free-text keywords to the dataset's actual article_type values. Keys
# match article_type exactly as they appear in wardrobe_scored.csv.
GARMENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Sarees": ["saree", "sari"],
    "Kurtas": ["kurta", "kurti"],
    "Kurta Sets": ["kurta set"],
    "Kurtis": ["kurti"],
    "Shirts": ["shirt"],
    "Tshirts": ["t-shirt", "tshirt", "tee"],
    "Tops": ["top"],
    "Tunics": ["tunic"],
    "Dresses": ["dress"],
    "Jumpsuit": ["jumpsuit"],
    "Jeans": ["jeans"],
    "Jeggings": ["jeggings"],
    "Trousers": ["trousers", "pants", "chinos"],
    "Track Pants": ["track pants", "trackpants", "joggers"],
    "Lounge Pants": ["lounge pants"],
    "Capris": ["capris"],
    "Churidar": ["churidar"],
    "Salwar": ["salwar"],
    "Salwar and Dupatta": ["salwar and dupatta", "salwar suit"],
    "Patiala": ["patiala"],
    "Shorts": ["shorts"],
    "Lounge Shorts": ["lounge shorts"],
    "Skirts": ["skirt"],
    "Leggings": ["leggings"],
    "Jackets": ["jacket"],
    "Rain Jacket": ["rain jacket", "raincoat"],
    "Sweaters": ["sweater", "jumper"],
    "Sweatshirts": ["sweatshirt", "hoodie"],
    "Tracksuits": ["tracksuit"],
    "Waistcoat": ["waistcoat", "vest"],
    "Shrug": ["shrug"],
    "Swimwear": ["swimwear", "swimsuit", "bikini"],
    "Nightdress": ["nightdress", "nightgown"],
    "Night suits": ["night suit", "nightsuit", "pyjama", "pajama"],
    "Bath Robe": ["bathrobe", "bath robe"],
}


@dataclass
class OutfitResult:
    """A single outfit recommendation and its justification.

    fallback_note is set whenever a constraint (season, formality, or the
    specifically requested garment type) had to be relaxed to produce this
    result callers should surface it to the user rather than let the
    relaxation pass unstated.

    matched_season records the season that was actually matched (if any),
    for display purposes; it is None when no season-specific match was found
    and the season constraint was relaxed.
    """

    items: list[dict[str, Any]]
    rationale: str
    source: str  # "rule_based" or "llm"
    total_reuse: int
    avg_sustainability: float
    fallback_note: str | None = None
    matched_season: str | None = None

# Free-text request parsing

def detect_garment_type(question: str) -> str | None:
    """Match free text to a specific article_type.

    Checks longer/more specific keyword phrases first, so e.g. "kurta set"
    is matched before the shorter "kurta" keyword would shadow it. Returns
    None if no known garment-type keyword appears in the question.
    """
    query = question.casefold()
    ordered_by_specificity = sorted(
        GARMENT_TYPE_KEYWORDS.items(),
        key=lambda pair: max(len(keyword) for keyword in pair[1]),
        reverse=True,
    )
    for article_type, keywords in ordered_by_specificity:
        if any(keyword in query for keyword in keywords):
            return article_type
    return None


def _article_type_display(article_type: str) -> str:
    """Singular, lowercase display form of an article_type for use in prose."""
    return ARTICLE_TYPE_SINGULAR.get(article_type, article_type.lower())


# Candidate-pool filtering (shared by both recommendation paths)

def _filter_to_formality(wardrobe: pd.DataFrame, event: str) -> pd.DataFrame:
    """Keep only garments whose usage/formality suits the occasion.

    Uses the dataset's `usage` field (Casual, Formal, Smart Casual, Ethnic,
    Sports). Falls back to the unfiltered set if the filter would leave
    nothing, so a sparse wardrobe still returns a recommendation.
    """
    if "usage" not in wardrobe.columns:
        return wardrobe
    allowed = EVENT_FORMALITY_MAP.get(event)
    if not allowed:
        return wardrobe
    usage = wardrobe["usage"].fillna("").astype(str).str.strip().str.casefold()
    matches = wardrobe[usage.isin(allowed)]
    return matches if not matches.empty else wardrobe


def _filter_to_season_preference(
    wardrobe: pd.DataFrame, event: str
) -> tuple[pd.DataFrame, str | None]:
    """Try each season in the event's own ordered preference list, in turn.

    Returns the first non-empty match together with which season produced
    it. Returns (empty DataFrame, None) only if none of the occasion's
    acceptable seasons match anything in `wardrobe` callers are then
    responsible for their own explicit, user-disclosed further fallback.
    """
    for season in EVENT_SEASON_MAP.get(event, DEFAULT_SEASONS):
        matches = wardrobe[wardrobe["season"].str.contains(season, case=False, na=False)]
        if not matches.empty:
            return matches, season
    return wardrobe.iloc[0:0], None


def _describe_event_seasons(event: str) -> str:
    """Human-readable list of the seasons considered acceptable for an occasion.

    Used only in user-facing fallback-note text (e.g. "...none of Summer,
    Spring, Fall was found...").
    """
    return ", ".join(EVENT_SEASON_MAP.get(event, DEFAULT_SEASONS))


def _apply_ranking_strategy(
    candidates: pd.DataFrame, style_first: bool
) -> tuple[pd.DataFrame, str]:
    """Prepare a candidate pool for ranking under the active strategy.

    Sustainability-first ranks directly by `sustainability_score`.
    Style-first ranks by ascending reuse count (i.e. "newest"/least-worn
    first), via a temporary `_style_rank` column on a copy of the input
    the real `sustainability_score` column is left untouched so it remains
    available for display and cross-strategy comparison.

    Returns the DataFrame to sort (copied only when necessary) and the
    column name to sort it by.
    """
    if not style_first:
        return candidates, "sustainability_score"
    ranked = candidates.copy()
    ranked["_style_rank"] = -ranked["reuse_count"]
    return ranked, "_style_rank"


def _select_outfit(
    season_items: pd.DataFrame,
    sort_column: str = "sustainability_score",
) -> list[pd.Series]:
    """Select the best top/dress + bottomwear combination, ranked by `sort_column`.

    Mirrors _select_outfit() in 08_generate_prompts.py. A dress or apparel
    set is treated as a complete one-piece outfit on its own; otherwise the
    best-ranked topwear item is paired with the best-ranked bottomwear item.
    If no topwear/bottomwear split is available at all, the single
    best-ranked item in the pool is returned so a recommendation is still
    possible from a sparse or unusually structured wardrobe.
    """
    outfit_items: list[pd.Series] = []

    tops = season_items[
        season_items["sub_category"].str.contains("Topwear|Dress", case=False, na=False)
    ]
    best_top = None
    if not tops.empty:
        best_top = tops.sort_values(by=sort_column, ascending=False).iloc[0]
        outfit_items.append(best_top)

    is_dress = best_top is not None and "Dress" in str(best_top.get("sub_category", ""))
    if best_top is None or not is_dress:
        bottoms = season_items[
            season_items["sub_category"].str.contains("Bottomwear", case=False, na=False)
        ]
        if not bottoms.empty:
            best_bottom = bottoms.sort_values(by=sort_column, ascending=False).iloc[0]
            outfit_items.append(best_bottom)

    if not outfit_items and not season_items.empty:
        outfit_items.append(season_items.sort_values(by=sort_column, ascending=False).iloc[0])

    return outfit_items

# Garment -> display-item conversion

def _series_to_item(row: pd.Series) -> dict[str, Any]:
    """Extract the display fields for one garment from its dataset row.

    Resolves the CO2e proxy from the current field name first
    (counterfactual_new_garment_production_co2e_kg), falling back to the
    older estimated_production_co2e_kg for backward compatibility with
    earlier pipeline runs. Returns None for co2e_kg (never a fabricated
    number) when neither field holds a usable value, consistent with the
    pipeline's no-fabrication policy.
    """
    raw_id = row.get("garment_id", "")
    garment_id = str(raw_id).strip()
    if garment_id.endswith(".0"):
        garment_id = garment_id[:-2]

    co2e_kg: float | None = None
    for column in (
        "counterfactual_new_garment_production_co2e_kg",
        "estimated_production_co2e_kg",
    ):
        value = row.get(column)
        if value is not None and pd.notna(value) and str(value).strip() != "":
            try:
                co2e_kg = float(value)
                break
            except (TypeError, ValueError):
                continue

    return {
        "garment_id": garment_id,
        "product_name": row.get("product_name", "Unknown item"),
        "gender": row.get("gender", ""),
        "material": row.get("extracted_material", row.get("primary_material", "")),
        "sub_category": row.get("sub_category", ""),
        "season": row.get("season", ""),
        "reuse_count": int(row.get("reuse_count", 0)),
        "sustainability_score": float(row.get("sustainability_score", 0.0)),
        "co2e_kg": co2e_kg,
    }


# Rationale-text helpers (shared by both recommendation paths)

def _aggregate_co2e(items: list[dict[str, Any]]) -> float | None:
    """Sum the CO2e production-impact proxy across a set of recommended items.

    Items with no available proxy are skipped rather than treated as zero
    (see _series_to_item), so the total reflects only garments with real
    evidence behind them. Works equally for a single-item or multi-item
    outfit, since summing a single value returns that value. Returns None
    if no item in the set has a usable value.
    """
    values = [item["co2e_kg"] for item in items if item.get("co2e_kg") is not None]
    return round(sum(values), 1) if values else None


def _co2_avoidance_sentence(total_co2: float | None, pronoun: str) -> str:
    """Build the optional "avoided production impact" rationale sentence.

    Returns an empty string when no proxy total is available, so callers can
    splice the result directly into a rationale string without a separate
    conditional at the call site. `pronoun` should be "this" for a
    single-garment recommendation or "these" for a multi-item outfit.
    """
    if total_co2 is None:
        return ""
    return (
        f" By reusing {pronoun} instead of buying new, you avoid an "
        f"estimated {total_co2} kg CO\u2082e of new-garment production."
    )


# 1. Rule-based recommendation (always available)

def _recommend_specific_type(
    wardrobe: pd.DataFrame,
    event: str,
    article_type: str,
    style_first: bool,
) -> OutfitResult:
    """Recommend the single best-scoring item of an explicitly requested type.

    Applies the same occasion/season narrowing and disclosed-fallback
    philosophy as the full-outfit path in rule_based_recommendation(), but
    never substitutes a different garment type for the one requested
    if the wardrobe genuinely has none, that is reported plainly rather
    than silently recommending something else.
    """
    display_name = _article_type_display(article_type)

    type_matches = wardrobe[wardrobe["article_type"] == article_type]
    if type_matches.empty:
        return OutfitResult(
            [], f"No {display_name} was found in this wardrobe.", "rule_based", 0, 0.0
        )

    working, sort_column = _apply_ranking_strategy(type_matches, style_first)

    fallback_note: str | None = None
    formality_matches = _filter_to_formality(working, event)
    season_matches, display_season = _filter_to_season_preference(formality_matches, event)

    if display_season is not None:
        pool = season_matches
    elif not formality_matches.empty:
        pool = formality_matches
        fallback_note = (
            f"No {display_name} tagged with any of this occasion's usual "
            f"seasons ({_describe_event_seasons(event)}) was found, so the "
            f"season constraint was relaxed."
        )
    else:
        pool = working
        fallback_note = (
            f"This wardrobe's {display_name} options did not match the "
            f"occasion's usual formality, so the closest available "
            f"{display_name} is shown instead."
        )

    best = pool.sort_values(by=sort_column, ascending=False).iloc[0]
    items = [_series_to_item(best)]
    total_reuse = items[0]["reuse_count"]
    avg_sust = items[0]["sustainability_score"]

    co2_sentence = ""
    if not style_first:
        co2_sentence = _co2_avoidance_sentence(_aggregate_co2e(items), pronoun="this")

    name = f"your {items[0]['product_name']} ({items[0]['material']})"
    if style_first:
        rationale = (
            f"For your {event}, I'd suggest {name}. This is among the "
            f"freshest, least-worn {display_name} in the wardrobe."
        )
    else:
        season_label = f" ({display_season})" if display_season else ""
        rationale = (
            f"For your {event}{season_label}, I recommend {name}, your "
            f"highest-scoring {display_name} for this occasion. It has "
            f"been worn {total_reuse} times, supporting continued reuse. "
            f"Sustainability score: {avg_sust}/10.{co2_sentence}"
        )

    return OutfitResult(
        items, rationale, "rule_based", total_reuse, avg_sust, fallback_note, display_season
    )


def rule_based_recommendation(
    wardrobe: pd.DataFrame,
    event: str,
    style_first: bool = False,
    requested_article_type: str | None = None,
) -> OutfitResult:
    """Recommend an outfit using the pipeline's own selection strategy.

    When style_first is True, items are chosen by lowest reuse count
    ("newest"), mirroring the style-first baseline in
    09_generate_baseline_prompts.py, so the app can demonstrate the
    sustainability-first vs. style-first contrast without any LLM.

    When requested_article_type is set, delegates to
    _recommend_specific_type() to return a single item of that type instead
    of a full top+bottom outfit.

    Any relaxed constraint (season, formality) along the way is reported via
    OutfitResult.fallback_note rather than silently substituted.
    """
    if wardrobe.empty:
        return OutfitResult(
            [], "This wardrobe has no scored garments to recommend from.", "rule_based", 0, 0.0
        )

    if requested_article_type:
        return _recommend_specific_type(wardrobe, event, requested_article_type, style_first)

    formality_items = _filter_to_formality(wardrobe, event)
    season_items, display_season = _filter_to_season_preference(formality_items, event)

    fallback_note: str | None = None
    if display_season is None:
        season_items = formality_items
        fallback_note = (
            f"No item tagged with any of this occasion's usual seasons "
            f"({_describe_event_seasons(event)}) was found, so the season "
            f"constraint was relaxed."
        )

    working, sort_column = _apply_ranking_strategy(season_items, style_first)
    selected = _select_outfit(working, sort_column=sort_column)

    # First further fallback: relax SEASON entirely (keep formality) if the
    # outfit is still incomplete e.g. a top exists but no bottom at all
    # within the season-matched pool.
    if len(selected) < 2 and len(formality_items) > 1:
        full, _ = _apply_ranking_strategy(formality_items, style_first)
        retry = _select_outfit(full, sort_column=sort_column)
        if len(retry) > len(selected):
            selected = retry
            if fallback_note is None:
                fallback_note = (
                    "This wardrobe did not have a complete occasion-appropriate "
                    "outfit, so the closest available combination is shown."
                )

    # Second further fallback: relax formality to the whole wardrobe, so the
    # outfit completes even if the wardrobe has no occasion-matched garment
    # of the missing piece at all. Formality is left uneven by design, so
    # this is the honest "closest available" behaviour rather than
    # fabricating a match.
    if len(selected) < 2 and len(wardrobe) > 1:
        full, _ = _apply_ranking_strategy(wardrobe, style_first)
        retry = _select_outfit(full, sort_column=sort_column)
        if len(retry) > len(selected):
            selected = retry
            fallback_note = (
                "This wardrobe did not have enough occasion-appropriate items, "
                "so the recommendation draws from the closest available garments "
                "in the wardrobe overall."
            )

    items = [_series_to_item(row) for row in selected]
    total_reuse = sum(item["reuse_count"] for item in items)
    avg_sust = (
        round(sum(item["sustainability_score"] for item in items) / len(items), 2)
        if items
        else 0.0
    )

    names = " paired with ".join(
        f"your {item['product_name']} ({item['material']})" for item in items
    )

    if style_first:
        rationale = (
            f"For your {event}, I'd suggest {names}. These are among the "
            f"freshest, least-worn pieces in the wardrobe, so they look "
            f"crisp and put-together for the occasion."
        )
    else:
        co2_sentence = _co2_avoidance_sentence(_aggregate_co2e(items), pronoun="these")
        season_label = f" ({display_season})" if display_season else ""
        rationale = (
            f"For your {event}{season_label}, I recommend {names}. This reuses "
            f"garments already owned rather than buying new, favours "
            f"lower-impact materials, and the pieces have been worn "
            f"{total_reuse} times combined \u2014 supporting lower-impact reuse. "
            f"Average sustainability score: {avg_sust}/10.{co2_sentence}"
        )

    return OutfitResult(
        items, rationale, "rule_based", total_reuse, avg_sust, fallback_note, display_season
    )

def ollama_available() -> bool:
    """True only if a local Ollama server is reachable right now."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
 
 
def _build_wardrobe_prompt(wardrobe: pd.DataFrame, event: str) -> str:
    """Render a wardrobe as the exact prompt format the model was fine-tuned
    and evaluated on (see src/08_generate_prompts.py), so its behaviour here
    matches what was measured in Chapter 5."""
    lines = [
        f"User Request: I have a {event} today. It is currently "
        f"{EVENT_SEASON_MAP.get(event, DEFAULT_SEASONS)[0]}. Please recommend a "
        f"complete, sustainable outfit from my wardrobe below that is "
        f"appropriate for the weather, rather than me buying something new.",
        "My Wardrobe:",
    ]
    for _, row in wardrobe.iterrows():
        material = row.get("extracted_material", row.get("primary_material", ""))
        lines.append(
            f"- {row.get('product_name', 'Unknown item')} "
            f"[Category: {row.get('sub_category', '')}, Season: {row.get('season', '')}] "
            f"(Material: {material}, Worn: {int(row.get('reuse_count', 0))} times, "
            f"Score: {row.get('sustainability_score', 0)})"
        )
    return "\n".join(lines)
 
 
def _match_items_in_response(response_text: str, wardrobe: pd.DataFrame) -> list[pd.Series]:
    """Identify which real wardrobe rows the model's generated text actually
    named, in the order they're mentioned. If multiple wardrobe rows share
    an identical product_name, only the first is kept per mention the
    generated text can't disambiguate which physical item was meant, and
    counting every same-named duplicate would misrepresent a single-item
    mention as multiple items. Returns an empty list if nothing could be
    confidently matched callers must handle this explicitly rather than
    assume a match always exists."""
    matches: list[tuple[int, pd.Series]] = []
    seen_names: set[str] = set()
    lower_text = response_text.lower()
    for _, row in wardrobe.iterrows():
        name = str(row.get("product_name", ""))
        key = name.lower()
        if name and key in lower_text and key not in seen_names:
            position = lower_text.find(key)
            matches.append((position, row))
            seen_names.add(key)
    matches.sort(key=lambda pair: pair[0])
    return [row for _, row in matches]
 
def ollama_recommendation(
    wardrobe: pd.DataFrame,
    event: str,
    model_name: str = OLLAMA_MODEL_NAME,
) -> OutfitResult:
    """Generate a recommendation using the actual fine-tuned model, running
    locally via Ollama. The model's own output determines both the garments
    recommended and the justification not a rule-based selection with
    LLM-authored prose, unlike llm_recommendation().

    Raises RuntimeError if Ollama is not reachable or the request fails
    callers should catch this and fall back to rule_based_recommendation().
    """
    if wardrobe.empty:
        return OutfitResult(
            [], "This wardrobe has no scored garments to recommend from.",
            "rule_based", 0, 0.0,
        )

    if not ollama_available():
        raise RuntimeError("Ollama is not reachable at localhost:11434.")

    prompt = _build_wardrobe_prompt(wardrobe, event)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "seed": 42,
                    "num_predict": 200,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        generated_text = response.json().get("response", "").strip()
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error

    if not generated_text:
        raise RuntimeError("Ollama returned an empty response.")

    matched_rows = _match_items_in_response(generated_text, wardrobe)

    fallback_note: str | None = None
    if not matched_rows:
        fallback_note = (
            "The fine-tuned model's response could not be automatically "
            "matched to specific wardrobe items shown as generated text only."
        )
        return OutfitResult(
            [], generated_text, "fine_tuned_local", 0, 0.0, fallback_note,
        )

    items = [_series_to_item(row) for row in matched_rows]
    total_reuse = sum(item["reuse_count"] for item in items)
    avg_sust = round(
        sum(item["sustainability_score"] for item in items) / len(items), 2
    )

    if len(items) < 2:
        fallback_note = (
            "The fine-tuned model's response named a single garment for this "
            "request, consistent with its evaluated 32% outfit-completeness "
            "rate on held-out prompts."
        )
    else:
        # Honest transparency check: does a higher-scoring combination exist
        # in this wardrobe than the one the model actually chose? This never
        # changes the model's own selection it only discloses the gap,
        # consistent with the app's existing disclosure principle and with
        best_possible = _select_outfit(wardrobe, sort_column="sustainability_score")
        if best_possible:
            best_possible_avg = round(
                sum(row["sustainability_score"] for row in best_possible) / len(best_possible),
                2,
            )
            if best_possible_avg > avg_sust:
                fallback_note = (
                    f"A higher-scoring combination exists in this wardrobe "
                    f"(avg {best_possible_avg}/10). The fine-tuned model's own "
                    f"selection is shown here, consistent with its evaluated "
                    f"26% exact-match rate against optimal targets."
                )

    return OutfitResult(
        items, generated_text, "fine_tuned_local", total_reuse, avg_sust, fallback_note,
    )
# Unified entry point used by the Streamlit app

def recommend(
    wardrobe: pd.DataFrame,
    event: str,
    question: str | None = None,
    style_first: bool = False,
    use_fine_tuned: bool = False,
    requested_article_type: str | None = None,
) -> tuple[OutfitResult, str | None]:
    """Return a recommendation, transparently falling back to rule-based.

    Two engines are available:
      - use_fine_tuned=True: the actual fine-tuned model (local, via Ollama)
        generates the full response, including which garments to recommend.
        Does not support style_first or requested_article_type only the
        sustainability-first model was fine-tuned and evaluated.
      - Otherwise: the rule-based engine handles everything, deterministically.

    Returns (result, warning). `warning` is a user-facing note when the
    fine-tuned engine had to fall back (e.g. Ollama not running), otherwise
    None.
    """
    if use_fine_tuned:
        try:
            return ollama_recommendation(wardrobe, event), None
        except Exception as error:  # noqa: BLE001 - surface any failure gracefully
            warning = (
                f"Fine-tuned model unavailable ({type(error).__name__}); showing "
                f"the rule-based recommendation instead."
            )
            return (
                rule_based_recommendation(wardrobe, event, style_first, requested_article_type),
                warning,
            )

    return rule_based_recommendation(wardrobe, event, style_first, requested_article_type), None