"""Tests for the scoring stages added after the initial material pipeline:

- 04d_score_blends.py            (renormalised weighted-average blend scoring)
- 06b_estimate_environmental_impact.py  (kg CO2e proxy)
- 06c_sensitivity_analysis.py    (weight sensitivity of the composite score)

These modules use numeric-prefixed filenames, so they are loaded via importlib
rather than a normal import.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

blends = importlib.import_module("src.04d_score_blends")
impact = importlib.import_module("src.06b_estimate_environmental_impact")
sensitivity = importlib.import_module("src.06c_sensitivity_analysis")


# -
# 1. Blend scoring
# -

def _components(pairs):
    """Build the composition_percentages_json structure the parser emits."""
    return json.dumps(
        [{"canonical_material": m, "percentage": p} for m, p in pairs]
    )


def test_valid_blend_receives_weighted_material_score():
    """A valid percentage blend receives the correct renormalised weighted score."""
    # cotton=5.0, polyester=4.0  ->  0.6*5 + 0.4*4 = 4.6
    comps = blends._parse_components(_components([("cotton", 60), ("polyester", 40)]))
    score, status, _ = blends._blend_score(comps)
    assert score == pytest.approx(4.6, abs=0.01)
    assert status == "scored_blend_weighted_average"


def test_blend_renormalises_over_benchmarked_components():
    """A component without a benchmark (elastane) is dropped and weights renormalise."""
    # 95% cotton + 5% elastane -> cotton only -> 5.0, flagged partial coverage
    comps = blends._parse_components(_components([("cotton", 95), ("elastane", 5)]))
    score, status, _ = blends._blend_score(comps)
    assert score == pytest.approx(5.0, abs=0.01)
    assert status == "scored_blend_weighted_average_partial_coverage"


def test_blend_with_no_benchmarked_component_remains_unscored():
    """A blend whose components all lack benchmarks is not scored (no default value)."""
    comps = blends._parse_components(_components([("elastane", 100)]))
    score, status, _ = blends._blend_score(comps)
    assert score is None
    assert status == "unscored_blend_no_benchmarked_component"


def test_blend_with_unparseable_composition_is_not_scored():
    """Invalid/empty composition JSON yields no components and no fabricated score."""
    comps = blends._parse_components("not valid json")
    assert comps == []
    score, status, _ = blends._blend_score(comps)
    assert score is None


# -
# 2. Environmental-impact proxy
# -

def test_scored_garment_receives_positive_co2e_estimate():
    """A garment with a known material and article type gets an approximate kg CO2e."""
    factor = impact._material_factor("cotton")
    weight = impact._garment_weight("Tshirts")
    assert factor is not None
    est = factor * weight
    assert est > 0
    # cotton tee should be a small, plausible figure (a few kg), not zero or huge
    assert 0.5 < est < 10


def test_missing_material_factor_does_not_become_zero_emissions():
    """A material with no CO2 factor returns None, not 0.0 (no silent zero-impact)."""
    assert impact._material_factor("unobtainium") is None


def test_unknown_article_type_uses_documented_default_weight():
    """An unlisted article type falls back to the documented default weight, not 0."""
    w = impact._garment_weight("some_unlisted_type")
    assert w == impact.DEFAULT_WEIGHT_KG
    assert w > 0


# -
# 3. Sensitivity analysis
# -

def _synthetic_scored(n=500):
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "material_score": rng.uniform(1, 8, n),
        "reuse_score": rng.uniform(0, 10, n),
        "wearability_score": rng.uniform(3, 10, n),
        "second_hand_bonus": rng.choice([0.0, 10.0], size=n),
    })


def test_all_four_scenarios_are_defined():
    assert set(sensitivity.SCENARIOS) == {
        "baseline", "material_focus", "reuse_focus", "equal_weights"
    }


def test_baseline_correlates_with_itself_at_one():
    """The baseline scenario must correlate perfectly with itself."""
    df = _synthetic_scored()
    baseline = sensitivity._weighted_score(df, sensitivity.SCENARIOS["baseline"])
    rho = baseline.corr(baseline, method="spearman")
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_weighted_score_stays_in_valid_range():
    """Re-weighted scores are clipped to the documented 1-10 range."""
    df = _synthetic_scored()
    for weights in sensitivity.SCENARIOS.values():
        s = sensitivity._weighted_score(df, weights)
        assert s.min() >= 1.0
        assert s.max() <= 10.0


def test_alternative_weighting_changes_scores_but_preserves_length():
    """A different weighting produces a different score series of the same length."""
    df = _synthetic_scored()
    base = sensitivity._weighted_score(df, sensitivity.SCENARIOS["baseline"])
    alt = sensitivity._weighted_score(df, sensitivity.SCENARIOS["reuse_focus"])
    assert len(base) == len(alt) == len(df)
    # They should not be identical for a non-trivial dataset.
    assert not base.equals(alt)


# -
# 4. Gender-coherent wardrobe generation
# -

wardrobe = importlib.import_module("src.05_generate_wardrobe_attributes")


def _gendered_frame(n=600):
    import numpy as np
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "gender": rng.choice(["Men", "Women", "Unisex"], size=n, p=[0.45, 0.45, 0.10]),
        "sub_category": rng.choice(["Topwear", "Bottomwear"], size=n),
    })


def test_owners_are_assigned_within_range():
    """Every assigned owner id is within the requested owner pool."""
    import numpy as np
    df = _gendered_frame()
    n_owners = 200
    owners = wardrobe._assign_gender_coherent_owners(
        df, np.random.default_rng(1), n_owners
    )
    assert owners.min() >= 1
    assert owners.max() <= n_owners
    assert len(owners) == len(df)


def test_no_wardrobe_mixes_mens_and_womens_garments():
    """Core coherence rule: an owner never holds both men's and women's items.

    Unisex items may appear in any wardrobe, but a men's garment and a
    women's garment must never share an owner.
    """
    import numpy as np
    df = _gendered_frame()
    df = df.copy()
    df["wardrobe_owner_id"] = wardrobe._assign_gender_coherent_owners(
        df, np.random.default_rng(1), 200
    )
    g = df["gender"].str.casefold()
    for _owner, grp in df.groupby("wardrobe_owner_id"):
        genders = set(grp["gender"].str.casefold()) - {"unisex"}
        assert not ("men" in genders and "women" in genders), (
            f"Owner {_owner} mixes men's and women's garments"
        )


def test_ensure_top_and_bottom_gives_every_owner_both():
    """After rebalancing, no owner is left without a top or without a bottom."""
    import numpy as np
    df = _gendered_frame()
    df = df.copy()
    df["wardrobe_owner_id"] = wardrobe._assign_gender_coherent_owners(
        df, np.random.default_rng(2), 200
    )
    df = wardrobe._ensure_top_and_bottom_per_owner(df, np.random.default_rng(3))
    sub = df["sub_category"].str.casefold()
    df2 = df.assign(
        _t=sub.str.contains("topwear|dress"),
        _b=sub.str.contains("bottomwear"),
    )
    counts = df2.groupby("wardrobe_owner_id")[["_t", "_b"]].sum()
    assert (counts["_t"] > 0).all(), "some owner has no top after rebalancing"
    assert (counts["_b"] > 0).all(), "some owner has no bottom after rebalancing"