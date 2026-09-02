"""test_json_material_evidence_audit.py

Tests for the JSON material-evidence extraction and classification pipeline
in src/json_material_evidence_audit.py. Each test constructs a minimal
API-shaped payload (mirroring the real `{"data": {...}}` wrapper used by the
raw JSON files) and checks the resulting classification.
"""

from __future__ import annotations

from src.json_material_evidence_audit import inspect_json_record

# NOTE: the original version of this file also imported
# normalise_composition_text, parse_material_text, and
# classify_material_evidence from `material_parser_core` (without the
# `src.` package prefix which would fail unless `src/` happened to already
# be on sys.path). None of those three names were actually used anywhere in
# this file, so the import has been removed rather than fixed. If you need
# to test material_parser_core directly, do that in
# test_material_parser_core.py instead of here.


def test_api_wrapper_is_unwrapped():
    payload = {
        "data": {
            "id": 1530,
            "ageGroup": "Adults-Men",
            "productDisplayName": "Puma Men Ferrari Track Jacket",
            "productDescriptors": {
                "description": {
                    "value": "<p>85% cotton and 15% polyester double knit jacket.</p>"
                }
            },
            "articleAttributes": {},
        }
    }

    record, components = inspect_json_record(garment_id="1530", payload=payload)

    assert record["json_id"] == "1530"
    assert record["primary_material"] == "cotton"
    assert record["secondary_materials"] == "polyester"
    assert record["composition_status"] == "exact_blend_percentages"
    assert record["scoring_eligibility"] == "known_blend_pending_policy"
    assert len(components) == 2


def test_detailed_percentage_overrides_fabric_dropdown():
    payload = {
        "data": {
            "id": 28690,
            "productDisplayName": "W Women Printed Beige Kurta",
            "productDescriptors": {
                "description": {
                    "value": "<p>Composition beige kurta made of 100% cotton.</p>"
                }
            },
            "articleAttributes": {"Fabric": "Dupion Silk"},
        }
    }

    record, _ = inspect_json_record(garment_id="28690", payload=payload)

    assert record["primary_material"] == "cotton"
    assert record["composition_status"] == "exact_single_fibre_percentage"
    assert record["source_conflict"] is True


def test_leather_shoes_are_not_garment_material():
    payload = {
        "data": {
            "id": 8834,
            "productDisplayName": "Scullers Men Brown Trousers",
            "productDescriptors": {
                "description": {
                    "value": (
                        "<p>Composition brown trousers with pockets. "
                        "Pair with leather shoes for a formal look.</p>"
                    )
                }
            },
            "articleAttributes": {"Fabric": "Cotton"},
        }
    }

    record, _ = inspect_json_record(garment_id="8834", payload=payload)

    assert record["primary_material"] == "cotton"
    assert "leather" not in record["secondary_materials"]
    assert record["composition_status"] == "single_fibre_claim_unquantified"


def test_lambswool_and_nylon_are_detected():
    payload = {
        "data": {
            "id": 19956,
            "productDisplayName": "Men Green Sweater",
            "productDescriptors": {
                "materials_care_desc": {
                    "value": "<p>80% lambswool, 20% nylon. Dry clean only.</p>"
                }
            },
            "articleAttributes": {"Fabric": "Wool"},
        }
    }

    record, components = inspect_json_record(garment_id="19956", payload=payload)

    assert record["primary_material"] == "wool"
    assert record["secondary_materials"] == "nylon"
    assert record["composition_status"] == "exact_blend_percentages"
    assert len(components) == 2


def test_malformed_percentage_comma_is_repaired():
    payload = {
        "data": {
            "id": 54332,
            "productDisplayName": "Garfield Women Briefs",
            "productDescriptors": {
                "materials_care_desc": {"value": "<p>84%, polyamide 16% elastane.</p>"}
            },
            "articleAttributes": {},
        }
    }

    record, components = inspect_json_record(garment_id="54332", payload=payload)

    assert record["primary_material"] == "nylon"
    assert record["secondary_materials"] == "elastane"
    assert record["percentage_total"] == 100.0
    assert record["composition_status"] == "exact_blend_percentages"
    assert len(components) == 2


def test_multi_part_bra_requires_role_review():
    payload = {
        "data": {
            "id": 50047,
            "productDisplayName": "Hanes Women Bra",
            "productDescriptors": {
                "materials_care_desc": {
                    "value": (
                        "<p>Cup and back lining: 82% nylon, 18% spandex. "
                        "Cup lining: 100% polyester.</p>"
                    )
                }
            },
            "articleAttributes": {"Fabric": "Cotton"},
        }
    }

    record, _ = inspect_json_record(garment_id="50047", payload=payload)

    assert record["composition_status"] == "role_specific_or_multi_part_composition"
    assert record["scoring_eligibility"] == "needs_role_review"


def test_styling_advice_does_not_add_leather_to_trouser_material():
    payload = {
        "data": {
            "id": 26166,
            "productDisplayName": "John Miller Men Striped Black Trousers",
            "productDescriptors": {
                "description": {
                    "value": (
                        "<p>Made of polyester and viscose blend. "
                        "Pair them with a crisp formal shirt and leather shoes.</p>"
                    )
                }
            },
            "articleAttributes": {},
        }
    }

    record, _ = inspect_json_record(garment_id="26166", payload=payload)

    # An unquantified blend ("polyester and viscose blend", no percentages)
    # deliberately has NO primary material the parser will not guess which
    # fibre dominates without percentages. This is the same conservative rule
    # verified by test_lycra_blend_is_not_treated_as_pure_elastane.
    assert record["primary_material"] == ""
    assert "polyester" in record["secondary_materials"]
    assert "viscose" in record["secondary_materials"]
    # The real point of this test: styling advice must not leak "leather" in.
    assert "leather" not in record["extracted_material"]
    assert record["composition_status"] == "named_blend_no_percentages"
    assert record["scoring_eligibility"] == "needs_review"


def test_combed_cotton_percentage_is_detected():
    payload = {
        "data": {
            "id": 17853,
            "productDisplayName": "Levis Men Boxer Grey Brief",
            "productDescriptors": {
                "materials_care_desc": {
                    "value": "<p>Made of 95% combed cotton and 5% elastane.</p>"
                }
            },
            "articleAttributes": {},
        }
    }

    record, components = inspect_json_record(garment_id="17853", payload=payload)

    assert record["primary_material"] == "cotton"
    assert record["secondary_materials"] == "elastane"
    assert record["percentage_total"] == 100.0
    assert record["composition_status"] == "exact_blend_percentages"
    assert record["scoring_eligibility"] == "known_blend_pending_policy"
    assert len(components) == 2


def test_trim_fabric_with_conflicting_totals_is_not_auto_scored():
    payload = {
        "data": {
            "id": 19333,
            "productDisplayName": "United Colors of Benetton Women Navy Blue Jacket",
            "productDescriptors": {
                "description": {
                    "value": (
                        "<p>Made of 100% polyester and trim fabric made of "
                        "97% polyester and 3% elastane.</p>"
                    )
                }
            },
            "articleAttributes": {},
        }
    }

    record, _ = inspect_json_record(garment_id="19333", payload=payload)

    # Two compositions in one string sum to 200%. The "trim" role IS detected,
    # but because the combined percentages do not form a valid single or
    # per-component 100% total, the parser conservatively classifies this as
    # partial/invalid rather than auto-scoring it. The key guarantee is that
    # it is NOT treated as a directly scorable single fibre.
    assert record["percentage_total"] == 200.0
    assert "trim" in record["material_role_flags"]
    assert record["composition_status"] == "partial_or_invalid_percentages"
    assert record["scoring_eligibility"] == "needs_percentage_review"
    assert record["scoring_eligibility"] != "eligible_exact_single_fibre"


def test_lycra_blend_is_not_treated_as_pure_elastane():
    payload = {
        "data": {
            "id": 16382,
            "productDisplayName": "Peter England Men Solid Grey Trouser",
            "productDescriptors": {
                "description": {
                    "value": (
                        "<p>Made of a lycra blend fabric with a flat front "
                        "and side pockets.</p>"
                    )
                }
            },
            "articleAttributes": {},
        }
    }

    record, _ = inspect_json_record(garment_id="16382", payload=payload)

    assert record["primary_material"] == ""
    assert record["secondary_materials"] == "elastane"
    assert record["composition_status"] == "named_blend_no_percentages"
    assert record["scoring_eligibility"] == "needs_review"