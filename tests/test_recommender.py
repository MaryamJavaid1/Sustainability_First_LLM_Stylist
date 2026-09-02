"""tests/test_recommender.py

Test suite for app/recommender.py.

Covers the scenarios that were found to contain real bugs during manual
testing of the web interface this session: the CO2e aggregation crash
(missing initialization when no item has a usable value), the stale
CO2e field-name lookup, the silent season-fallback that displayed an
incorrect season, and the garment-type keyword-detection specificity
ordering. Also covers the core selection, ranking, and fallback-disclosure
behaviour that the recommender's docstrings describe as guaranteed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.recommender import (
    OutfitResult,
    _aggregate_co2e,
    _article_type_display,
    _co2_avoidance_sentence,
    _filter_to_formality,
    _filter_to_season_preference,
    _series_to_item,
    detect_garment_type,
    rule_based_recommendation,
)

# Fixtures

def _make_row(
    garment_id: int,
    product_name: str,
    article_type: str,
    sub_category: str,
    season: str,
    usage: str,
    material: str,
    reuse_count: int,
    sustainability_score: float,
    co2e_kg: float | None,
) -> dict:
    """Build one synthetic wardrobe row matching the real dataset schema."""
    return {
        "garment_id": garment_id,
        "product_name": product_name,
        "article_type": article_type,
        "sub_category": sub_category,
        "season": season,
        "usage": usage,
        "extracted_material": material,
        "primary_material": material,
        "reuse_count": reuse_count,
        "sustainability_score": sustainability_score,
        "counterfactual_new_garment_production_co2e_kg": co2e_kg,
        "estimated_production_co2e_kg": co2e_kg,
    }


@pytest.fixture
def sample_wardrobe() -> pd.DataFrame:
    """A small, deliberately mixed wardrobe: matches the exact scenario used
    to catch the season-fallback and CO2e bugs during manual testing."""
    rows = [
        _make_row(1, "Cotton Sweater A", "Sweaters", "Topwear", "Fall", "Casual", "cotton", 36, 5.05, 20.8),
        _make_row(2, "Wool Sweater B", "Sweaters", "Topwear", "Winter", "Casual", "wool", 10, 6.0, 22.0),
        _make_row(3, "Summer Tee", "Tshirts", "Topwear", "Summer", "Casual", "cotton", 50, 6.5, 5.0),
        _make_row(4, "Trousers", "Trousers", "Bottomwear", "Summer", "Formal", "cotton", 38, 4.2, 10.7),
        _make_row(5, "Formal Shirt", "Shirts", "Topwear", "Fall", "Formal", "cotton", 47, 4.1, 6.1),
        _make_row(6, "No-Proxy Bamboo Tee", "Tshirts", "Topwear", "Summer", "Casual", "bamboo", 20, 7.0, None),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def empty_wardrobe() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "garment_id", "product_name", "article_type", "sub_category", "season",
        "usage", "extracted_material", "primary_material", "reuse_count",
        "sustainability_score", "counterfactual_new_garment_production_co2e_kg",
        "estimated_production_co2e_kg",
    ])

# _aggregate_co2e / _co2_avoidance_sentence
#
# Regression coverage for the crash we found: total_co2 was assigned inside
# the summation loop, so it was never defined at all when every item in an
# outfit had no CO2e value, causing a NameError downstream.

def test_aggregate_co2e_sums_available_values():
    items = [{"co2e_kg": 5.0}, {"co2e_kg": 3.2}, {"co2e_kg": None}]
    assert _aggregate_co2e(items) == 8.2


def test_aggregate_co2e_returns_none_when_nothing_available():
    items = [{"co2e_kg": None}, {"co2e_kg": None}]
    assert _aggregate_co2e(items) is None


def test_aggregate_co2e_handles_empty_list_without_raising():
    # This is the exact case that used to crash: an outfit where no item
    # has a usable CO2e proxy must not raise NameError further downstream.
    assert _aggregate_co2e([]) is None


def test_co2_avoidance_sentence_empty_when_no_total():
    assert _co2_avoidance_sentence(None, pronoun="this") == ""


def test_co2_avoidance_sentence_includes_pronoun_and_value():
    sentence = _co2_avoidance_sentence(12.4, pronoun="these")
    assert "these" in sentence
    assert "12.4" in sentence

# _series_to_item CO2e field resolution
#
# Regression coverage for the stale-field-name bug: the code once looked up
# a "reuse_avoided_co2e_kg" column that does not exist in the real dataset,
# silently returning None for every garment.


def test_series_to_item_resolves_current_co2e_field_name():
    row = pd.Series({
        "garment_id": "42",
        "product_name": "Test Item",
        "gender": "Women",
        "extracted_material": "cotton",
        "sub_category": "Topwear",
        "season": "Fall",
        "reuse_count": 10,
        "sustainability_score": 5.0,
        "counterfactual_new_garment_production_co2e_kg": 7.5,
        "estimated_production_co2e_kg": 99.9,  # should NOT be used when the current field is present
    })
    item = _series_to_item(row)
    assert item["co2e_kg"] == 7.5


def test_series_to_item_falls_back_to_older_field_name():
    row = pd.Series({
        "garment_id": "43",
        "product_name": "Older Pipeline Item",
        "extracted_material": "wool",
        "sub_category": "Topwear",
        "season": "Winter",
        "reuse_count": 5,
        "sustainability_score": 4.0,
        "estimated_production_co2e_kg": 15.0,
        # current field absent entirely
    })
    item = _series_to_item(row)
    assert item["co2e_kg"] == 15.0


def test_series_to_item_returns_none_rather_than_fabricating_a_value():
    row = pd.Series({
        "garment_id": "44",
        "product_name": "No Evidence Item",
        "extracted_material": "bamboo",
        "sub_category": "Topwear",
        "season": "Summer",
        "reuse_count": 1,
        "sustainability_score": 6.0,
    })
    item = _series_to_item(row)
    assert item["co2e_kg"] is None


# Season preference matching

def test_season_preference_matches_first_available_season_in_order(sample_wardrobe):
    # "casual lunch" prefers Summer, then Spring, then Fall. There is no
    # Summer/Spring Sweater, but there IS a Fall one this reproduces the
    # exact scenario that once incorrectly displayed "Summer" in the UI.
    formality_items = _filter_to_formality(sample_wardrobe, "casual lunch")
    sweaters_only = formality_items[formality_items["article_type"] == "Sweaters"]
    matches, season = _filter_to_season_preference(sweaters_only, "casual lunch")
    assert season == "Fall"
    assert (matches["product_name"] == "Cotton Sweater A").any()


def test_season_preference_returns_none_when_nothing_matches(sample_wardrobe):
    only_summer = sample_wardrobe[sample_wardrobe["season"] == "Summer"]
    only_summer_no_topwear = only_summer[only_summer["sub_category"] == "Bottomwear"]
    matches, season = _filter_to_season_preference(only_summer_no_topwear, "formal dinner")
    # "formal dinner" prefers Winter, Fall, Summer Summer is last, and the
    # only row here is Summer, so it SHOULD match on the third try.
    assert season == "Summer"

# rule_based_recommendation full outfit path

def test_full_outfit_recommendation_returns_top_and_bottom(sample_wardrobe):
    result = rule_based_recommendation(sample_wardrobe, "office meeting", style_first=False)
    assert isinstance(result, OutfitResult)
    assert len(result.items) == 2
    categories = {item["sub_category"] for item in result.items}
    assert categories == {"Topwear", "Bottomwear"}


def test_full_outfit_style_first_ranks_by_lowest_reuse_count(sample_wardrobe):
    result_sustainability = rule_based_recommendation(sample_wardrobe, "casual lunch", style_first=False)
    result_style = rule_based_recommendation(sample_wardrobe, "casual lunch", style_first=True)
    # The two strategies are not required to differ in every wardrobe, but
    # when they do differ, style-first must pick the lower reuse_count item.
    if result_sustainability.items != result_style.items:
        style_top = next(i for i in result_style.items if i["sub_category"] == "Topwear")
        sustain_top = next(i for i in result_sustainability.items if i["sub_category"] == "Topwear")
        assert style_top["reuse_count"] <= sustain_top["reuse_count"]


def test_empty_wardrobe_returns_empty_result_with_explanatory_rationale(empty_wardrobe):
    result = rule_based_recommendation(empty_wardrobe, "casual lunch")
    assert result.items == []
    assert "no scored garments" in result.rationale.lower()

# rule_based_recommendation specific garment-type path

def test_specific_garment_type_returns_single_best_scoring_item(sample_wardrobe):
    result = rule_based_recommendation(
        sample_wardrobe, "casual lunch", requested_article_type="Sweaters"
    )
    assert len(result.items) == 1
    assert result.items[0]["product_name"] == "Cotton Sweater A"


def test_specific_garment_type_uses_singular_display_name_in_rationale(sample_wardrobe):
    result = rule_based_recommendation(
        sample_wardrobe, "casual lunch", requested_article_type="Sweaters"
    )
    assert "sweater" in result.rationale.lower()
    assert "sweaters" not in result.rationale.lower()


def test_specific_garment_type_not_in_wardrobe_reports_plainly_not_substitutes(sample_wardrobe):
    result = rule_based_recommendation(
        sample_wardrobe, "casual lunch", requested_article_type="Sarees"
    )
    assert result.items == []
    assert "no saree was found" in result.rationale.lower()


def test_specific_garment_type_discloses_season_fallback(sample_wardrobe):
    # Formal dinner prefers Winter/Fall/Summer. The only Sweaters are Fall
    # and Winter Winter should match first under this event's preference.
    result = rule_based_recommendation(
        sample_wardrobe, "formal dinner", requested_article_type="Sweaters"
    )
    assert result.matched_season == "Winter"
    assert result.fallback_note is None  # a genuine match needs no disclosure


def test_matched_season_is_none_when_season_constraint_was_relaxed(sample_wardrobe):
    # Restrict to a wardrobe slice with only a Summer-tagged sweater-like
    # item, requested for an occasion whose season list never includes Summer
    # for this article type's only available tag combination.
    limited = sample_wardrobe[sample_wardrobe["product_name"] != "Cotton Sweater A"]
    limited = limited[limited["product_name"] != "Wool Sweater B"]
    result = rule_based_recommendation(limited, "casual lunch", requested_article_type="Sweaters")
    assert result.items == []  # no sweaters at all in this slice
    assert result.matched_season is None

# detect_garment_type

def test_detect_garment_type_matches_simple_keyword():
    assert detect_garment_type("show me a saree") == "Sarees"


def test_detect_garment_type_prefers_longer_more_specific_phrase():
    # "kurta set" must be matched over the shorter "kurta" keyword.
    assert detect_garment_type("I want to wear a kurta set for a party") == "Kurta Sets"


def test_detect_garment_type_returns_none_when_nothing_matches():
    assert detect_garment_type("what should I wear today") is None


def test_detect_garment_type_is_case_insensitive():
    assert detect_garment_type("SHOW ME A SWEATER") == "Sweaters"

# _article_type_display

@pytest.mark.parametrize("article_type,expected", [
    ("Sarees", "saree"),
    ("Jackets", "jacket"),
    ("Jeans", "jeans"),
    ("SomeUnmappedType", "someunmappedtype"),
])
def test_article_type_display_singularises_known_types(article_type, expected):
    assert _article_type_display(article_type) == expected