"""test_material_parser_core.py

Tests for the core material-composition parser in
src/material_parser_core.py.

FIX APPLIED: the original file had
    from material_parser_core import (...)
as its very but Python executes top-level statements in order, so that
import ran *before* the path fix existed. This only worked at all if `src/`
happened to already be importable some other way (e.g. running pytest from
inside `src/`, or an editable install). Running `pytest` from the project
root with a clean environment would raise `ModuleNotFoundError`.

The fix: import via the `src.` package path, consistent with every other
module in this project, and drop the sys.path manipulation entirely. This
requires `src/` to be importable as a package from the project root see
the note at the bottom of this file if that's not already the case for you.
"""

from __future__ import annotations

import unittest

from src.material_parser_core import (
    classify_material_evidence,
    parse_material_text,
)


def get_result(text: str) -> dict:
    parsed = parse_material_text("description", text)
    return classify_material_evidence(parsed)


class TestMaterialParserV2Core(unittest.TestCase):

    def check_status(self, text: str, expected_status: str):
        result = get_result(text)
        self.assertEqual(
            result["composition_status"],
            expected_status,
            msg=(
                f"\nInput: {text}"
                f"\nExpected: {expected_status}"
                f"\nGot: {result['composition_status']}"
                f"\nFull result: {result}"
            ),
        )

    def test_single_alias(self):
        self.check_status("95% organic cotton and 5% elasthen", "exact_blend_percentages")

    def test_modifier_before_material(self):
        self.check_status("95% super combed cotton and 5% elastane", "exact_blend_percentages")

    def test_malformed_percentage_punctuation(self):
        self.check_status("98.% cotton and 2% spandex", "exact_blend_percentages")

    def test_multi_part_shell_and_lining(self):
        self.check_status(
            "shell: 95% polyester and 5% spandex; lining: 100% polyester",
            "role_specific_or_multi_part_composition",
        )

    def test_set_with_two_components(self):
        self.check_status(
            "t-shirt: 100% cotton; bottom: 60% cotton and 40% polyester",
            "role_specific_or_multi_part_composition",
        )

    def test_unknown_material_with_percentages(self):
        self.check_status(
            "fabric - 80% tactel, 20% elastane foam cups - 100% polyester",
            "partial_or_invalid_percentages",
        )

    def test_missing_percent_symbol(self):
        self.check_status("97% cotton and 3 spandex", "partial_or_invalid_percentages")

    def test_single_named_blend_without_percentages(self):
        self.check_status("cotton blend", "named_blend_no_percentages")

    def test_two_material_named_blend_without_percentages(self):
        self.check_status("cotton linen blend", "named_blend_no_percentages")

    def test_repeated_single_fibre_statement(self):
        self.check_status(
            "made of 100% cotton, composition shirt made of 100% cotton",
            "exact_single_fibre_percentage",
        )

    def test_repeated_blend_statement(self):
        self.check_status(
            (
                "made of 98% cotton and 2% spandex, composition jacket "
                "made of 98% cotton and 2% spandex"
            ),
            "exact_blend_percentages",
        )

    def test_adjacent_percentages_without_separator(self):
        self.check_status(
            "lace-92% nylon 8% spandex and body-85% nylon 15% spandex",
            "role_specific_or_multi_part_composition",
        )

    def test_percentage_followed_by_lining_role(self):
        self.check_status(
            "55% polyester and 45% wool, 100% polyester lining",
            "role_specific_or_multi_part_composition",
        )

    def test_body_friendly_is_not_a_component(self):
        self.check_status(
            "100% polyester, body-friendly design with an elasticated waist",
            "exact_single_fibre_percentage",
        )

    def test_pyjama_set_is_multi_part(self):
        self.check_status(
            "t-shirt: 100% cotton; pyjama: 60% cotton and 40% polyester",
            "role_specific_or_multi_part_composition",
        )

    def test_percentage_followed_by_care_text(self):
        self.check_status(
            "100% cotton machine wash in 30 degrees with like colours",
            "exact_single_fibre_percentage",
        )

    def test_unknown_material_remains_partial(self):
        self.check_status("80% tactel, 20% elastane", "partial_or_invalid_percentages")

    def test_narrative_words_are_not_component_roles(self):
        self.check_status(
            (
                "100% cotton machine wash with like colours lace detailing "
                "on the yoke and a body-friendly fit"
            ),
            "exact_single_fibre_percentage",
        )

    def test_discount_percentage_is_not_material_evidence(self):
        self.check_status(
            (
                "100% cotton machine wash. Warranty valid for six months. "
                "Not valid on products with more than 20% discount."
            ),
            "exact_single_fibre_percentage",
        )

    def test_body_and_rib_are_separate_components(self):
        self.check_status(
            "body made of polyester fleece with rib made of 98% cotton and 2% elastane",
            "role_specific_or_multi_part_composition",
        )

    def test_track_suit_is_multi_part(self):
        self.check_status(
            "track jacket made of 100% polyester; track pant made of 100% polyester",
            "role_specific_or_multi_part_composition",
        )

    def test_lace_component_is_multi_part(self):
        self.check_status(
            "100% viscose top with lace made of 100% nylon",
            "role_specific_or_multi_part_composition",
        )

    def test_lambswool_alias(self):
        self.check_status("80% lambswool and 20% nylon", "exact_blend_percentages")

    def test_bamboo_alias(self):
        self.check_status("55% cotton and 45% bamboo", "exact_blend_percentages")

    def test_misspelled_elastane_alias(self):
        self.check_status("80% polyamide and 29% elstane", "partial_or_invalid_percentages")

    def test_down_and_feather_components(self):
        self.check_status(
            (
                "outer body made of 100% polyester, lining made of 100% polyester, "
                "filling made of 75% down and 25% feathers"
            ),
            "role_specific_or_multi_part_composition",
        )

    def test_ordinary_tshirt_phrase_is_not_a_component(self):
        self.check_status(
            "this t-shirt is made of 100% cotton and has short sleeves",
            "exact_single_fibre_percentage",
        )

    def test_explicit_tshirt_label_is_a_component(self):
        self.check_status(
            "t-shirt: 100% cotton; bottom: 60% cotton and 40% polyester",
            "role_specific_or_multi_part_composition",
        )

    def test_angora_alias(self):
        self.check_status(
            "60% cotton, 20% nylon, 10% angora and 10% wool",
            "exact_blend_percentages",
        )

    def test_terylene_alias(self):
        self.check_status("terylene and rayon blend", "named_blend_no_percentages")

    def test_polyster_alias(self):
        self.check_status("60% cotton and 40% polyster", "exact_blend_percentages")

    def test_spondac_alias(self):
        self.check_status("98.8% cotton and 1.2% spondac", "exact_blend_percentages")

    def test_structural_trim_is_multi_part(self):
        self.check_status(
            "body 56% viscose and 44% cotton and trim 100% cotton",
            "role_specific_or_multi_part_composition",
        )

    def test_ribbed_hem_is_not_a_component(self):
        self.check_status(
            "made of 100% cotton, has long sleeves with ribbed cuffs and a ribbed hem",
            "exact_single_fibre_percentage",
        )

    def test_single_component_with_lining_word(self):
        self.check_status(
            "100% polyester top with adjustable spaghetti straps",
            "exact_single_fibre_percentage",
        )

    def test_elastan_alias(self):
        self.check_status("93% polyamide and 7% elastan", "exact_blend_percentages")

    def test_set_with_t_shirt_and_legging(self):
        self.check_status(
            "t-shirt: 100% cotton; legging: 90% cotton and 10% spandex",
            "role_specific_or_multi_part_composition",
        )

    def test_body_and_bib_are_components(self):
        self.check_status(
            "100% viscose body with a ruffled bib made of 100% silk",
            "role_specific_or_multi_part_composition",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


# If `from src.material_parser_core import ...` fails with
# ModuleNotFoundError when you run `pytest` from the project root:
#
# Easiest fix add a pyproject.toml (or pytest.ini) at the project root with:
#
#     [tool.pytest.ini_options]
#     pythonpath = ["."]
#
# This tells pytest to add the project root to sys.path, so `src` resolves
# as a package. Also make sure `src/__init__.py` and `tests/__init__.py`
# exist (can be empty files) if you're not already using them.
