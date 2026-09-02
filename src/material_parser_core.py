"""material_parser_core.py

Core parser for garment material-composition evidence.

The parser separates quantified compositions, repeated statements, structural
or set-component compositions, malformed expressions, unknown materials, and
unquantified material claims. It does not assign LCA scores that happens in
04_apply_lca_scores.py.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

# Keep this dictionary as the single source of truth for recognised materials.
# Values are canonical materials used by the scoring and audit pipeline.
MATERIAL_ALIASES: dict[str, str] = {
    "organic cotton": "cotton",
    "recycled cotton": "cotton",
    "cotton": "cotton",
    "recycled polyester": "polyester",
    "terylene": "polyester",
    "polyster": "polyester",
    "polyester": "polyester",
    "pes": "polyester",
    "elasthane": "elastane",
    "elasthen": "elastane",
    "elastane": "elastane",
    "elastan": "elastane",
    "elstane": "elastane",
    "elastic": "elastane",
    "elastine": "elastane",
    "spondac": "elastane",
    "spandex": "elastane",
    "lycra": "elastane",
    "recycled nylon": "nylon",
    "recycled polyamide": "nylon",
    "polyamide": "nylon",
    "mylon": "nylon",
    "nylon": "nylon",
    "viscose": "viscose",
    "rayon": "viscose",
    "modal": "modal",
    "tencel": "lyocell",
    "lyocell": "lyocell",
    "linen": "linen",
    "flax": "linen",
    "ramie": "ramie",
    "angora": "wool",
    "lambswool": "wool",
    "merino wool": "wool",
    "wool": "wool",
    "cashmere": "cashmere",
    "silk": "silk",
    "hemp": "hemp",
    "jute": "jute",
    "bamboo": "bamboo",
    "acrylic": "acrylic",
    "acetate": "acetate",
    "cupro": "cupro",
    "polyurethane": "polyurethane",
    "leather": "leather",
    "down": "down",
    "feathers": "feather",
    "feather": "feather",
    # Retained because it occurs in the source data. It is accepted only in
    # composition-oriented text by the JSON audit script.
    "poly": "polyester",
}

# Canonical component labels. Labels are only accepted in explicit composition
# syntax, which prevents phrases such as "body-friendly" from becoming roles.
ROLE_ALIASES: dict[str, str] = {
    "outer shell": "shell",
    "main fabric": "main",
    "main garment": "main",
    "foam cups": "cup_pad",
    "foam cup": "cup_pad",
    "cup pad": "cup_pad",
    "cup lining": "cup_lining",
    "upper lining": "upper_lining",
    "lower lining": "lower_lining",
    "pocket bag": "pocket_bag",
    "front panel": "front_panel",
    "back panel": "back_panel",
    "side frame": "side_frame",
    "centre belt": "centre_belt",
    "center belt": "centre_belt",
    "upper body": "upper_body",
    "lower body": "lower_body",
    "track jacket": "track_jacket",
    "track pant": "track_pant",
    "track pants": "track_pant",
    "track suit": "tracksuit",
    "tracksuit": "tracksuit",
    "t-shirt": "t_shirt",
    "tshirt": "t_shirt",
    "tee shirt": "t_shirt",
    "leggings": "legging",
    "legging": "legging",
    "pyjama": "pyjama",
    "pajama": "pyjama",
    "bottom": "bottom",
    "shell": "shell",
    "outer": "outer",
    "body": "body",
    "bib": "bib",
    "lining": "lining",
    "lace": "lace",
    "trim": "trim",
    "trims": "trim",
    "trimmings": "trim",
    "ribbing": "ribbing",
    "ribs": "ribbing",
    "rib": "ribbing",
    "hood": "hood",
    "cup": "cup",
    "gusset": "gusset",
    "strap": "strap",
    "back": "back",
    "pad": "cup_pad",
    "panel": "panel",
    "insulation": "insulation",
    "filling": "filling",
    "yoke": "yoke",
    "hem": "hem",
}

# A direct label without punctuation is only safe for structural components.
DIRECT_PERCENT_ROLES = {
    "body", "bib", "shell", "outer", "lining", "lace", "trim",
    "rib", "ribbing", "gusset", "cup", "insulation", "filling",
    "front panel", "back panel", "upper body", "lower body", "yoke",
}

NON_MATERIAL_PERCENT_TERMS = {
    "discount", "off", "month", "months", "degree", "degrees", "gsm",
    "product", "products", "sale", "saving", "savings",
}

_ALIAS_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(MATERIAL_ALIASES, key=len, reverse=True)
)
_ROLE_PATTERN = "|".join(
    re.escape(role) for role in sorted(ROLE_ALIASES, key=len, reverse=True)
)
_DIRECT_ROLE_PATTERN = "|".join(
    re.escape(role) for role in sorted(DIRECT_PERCENT_ROLES, key=len, reverse=True)
)

MATERIAL_RE = re.compile(rf"\b(?P<material>{_ALIAS_PATTERN})\b", re.IGNORECASE)
PERCENTAGE_RE = re.compile(
    r"(?P<percentage>\d+(?:\.\d+)?)(?:\.)?\s*%",
    re.IGNORECASE,
)
MISSING_PERCENT_RE = re.compile(
    rf"\b\d+(?:\.\d+)?\s+(?:{_ALIAS_PATTERN})\b",
    re.IGNORECASE,
)
ROLE_HEADER_RE = re.compile(
    rf"\b(?P<role>{_ROLE_PATTERN})\b\s*(?::|-)\s*"
    r"(?=\d+(?:\.\d+)?(?:\.)?\s*%)",
    re.IGNORECASE,
)
ROLE_DIRECT_PERCENT_RE = re.compile(
    rf"\b(?P<role>{_DIRECT_ROLE_PATTERN})\b\s+"
    r"(?=\d+(?:\.\d+)?(?:\.)?\s*%)",
    re.IGNORECASE,
)
ROLE_MADE_OF_RE = re.compile(
    rf"\b(?P<role>{_ROLE_PATTERN})\b\s+made\s+(?:of|from)\s*"
    rf"(?=\d+(?:\.\d+)?(?:\.)?\s*%|(?:{_ALIAS_PATTERN})\b)",
    re.IGNORECASE,
)
ROLE_AND_HEADER_RE = re.compile(
    rf"\b(?P<role_one>{_ROLE_PATTERN})\b\s+and\s+"
    rf"(?P<role_two>{_ROLE_PATTERN})\b\s*:\s*"
    r"(?=\d+(?:\.\d+)?(?:\.)?\s*%)",
    re.IGNORECASE,
)
# A suffix role is valid only when it appears immediately after the material
# name, for example: "100% polyester lining". This prevents narrative text
# such as "ribbed hem" from being treated as a garment-component role.
_SUFFIX_ROLE_NAMES = (
    "upper lining", "lower lining", "pocket bag", "lining", "shell", "outer",
    "body", "trim", "ribbing", "rib", "hood", "cup", "gusset", "strap", "panel",
)
_SUFFIX_ROLE_PATTERN = "|".join(
    re.escape(role) for role in sorted(_SUFFIX_ROLE_NAMES, key=len, reverse=True)
)
SUFFIX_ROLE_RE = re.compile(
    rf"^\s*(?:{_ALIAS_PATTERN})\s+"
    rf"(?P<role>{_SUFFIX_ROLE_PATTERN})\b"
    r"\s*(?:[,.;]|$)",
    re.IGNORECASE,
)


def unique_preserve_order(values: list[str]) -> list[str]:
    """Return non-empty values once, preserving first appearance."""
    return list(dict.fromkeys(value for value in values if value))


def normalise_text(text: str | None) -> str:
    """Normalise case, whitespace, dash variants, and 98.% punctuation."""
    if not text:
        return ""

    cleaned = str(text).casefold().strip()
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    cleaned = re.sub(r"(\d+)\.\s*%", r"\1%", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def canonical_material(raw_material: str) -> str:
    """Return the canonical material for an already matched alias."""
    key = re.sub(r"\s+", " ", raw_material.casefold()).strip()
    return MATERIAL_ALIASES[key]


def find_materials(text: str) -> list[str]:
    """Return recognised canonical materials in first-seen order."""
    return unique_preserve_order(
        [canonical_material(match.group("material")) for match in MATERIAL_RE.finditer(text)]
    )


def _add_role_span(spans: list[tuple[int, str]], start: int, raw_role: str) -> None:
    role = ROLE_ALIASES[raw_role.casefold().strip()]
    if (start, role) not in spans:
        spans.append((start, role))


def extract_role_spans(text: str) -> list[tuple[int, str]]:
    """Find explicit role labels and their starting positions."""
    spans: list[tuple[int, str]] = []

    for match in ROLE_HEADER_RE.finditer(text):
        _add_role_span(spans, match.start(), match.group("role"))

    for match in ROLE_DIRECT_PERCENT_RE.finditer(text):
        _add_role_span(spans, match.start(), match.group("role"))

    for match in ROLE_MADE_OF_RE.finditer(text):
        _add_role_span(spans, match.start(), match.group("role"))

    for match in ROLE_AND_HEADER_RE.finditer(text):
        _add_role_span(spans, match.start(), match.group("role_one"))
        _add_role_span(spans, match.start(), match.group("role_two"))

    return sorted(spans, key=lambda span: span[0])


def extract_roles(text: str) -> list[str]:
    """Return explicit component labels in source order."""
    return unique_preserve_order([role for _, role in extract_role_spans(text)])


def role_for_percentage(
    text: str,
    percentage_start: int,
    fragment: str,
    role_spans: list[tuple[int, str]],
) -> str:
    """Assign a percentage to its nearest preceding explicit role label."""
    preceding = [span for span in role_spans if span[0] <= percentage_start]
    if preceding:
        return preceding[-1][1]

    suffix = SUFFIX_ROLE_RE.search(fragment)
    if suffix:
        return ROLE_ALIASES[suffix.group("role").casefold()]

    return "main"


def extract_percentage_pairs(text: str) -> tuple[list[dict[str, Any]], bool]:
    """Extract percentage-material pairs without treating discounts as material data."""
    matches = list(PERCENTAGE_RE.finditer(text))
    role_spans = extract_role_spans(text)
    pairs: list[dict[str, Any]] = []
    ignored_non_material_percentage = False

    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fragment = text[match.end():next_start].strip(" ,;:/-")
        first_word_match = re.match(r"[a-z]+", fragment)
        first_word = first_word_match.group(0) if first_word_match else ""

        # Skip whole fragments that begin with discount or care metadata,
        # even when a later word happens to be a recognised material.
        if first_word in NON_MATERIAL_PERCENT_TERMS:
            ignored_non_material_percentage = True
            continue

        material_match = MATERIAL_RE.search(fragment)

        if material_match:
            raw_material = material_match.group("material")
            canonical = canonical_material(raw_material)
            recognised = True
        else:
            raw_material = re.split(
                r"[,;]|\band\b|\bwith\b|\bbut\b",
                fragment,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            first_word_match = re.match(r"[a-z]+", raw_material)
            first_word = first_word_match.group(0) if first_word_match else ""
            if first_word in NON_MATERIAL_PERCENT_TERMS:
                ignored_non_material_percentage = True
                continue
            canonical = None
            recognised = False

        role = role_for_percentage(text, match.start(), fragment, role_spans)
        pairs.append(
            {
                "percentage": float(match.group("percentage")),
                "raw_material_text": raw_material,
                "material": canonical,
                "material_recognised": recognised,
                "role": role,
            }
        )

    return pairs, ignored_non_material_percentage


def unique_percentage_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove repeated composition statements for a non-multi-part garment.

    This function is called only after true structural and multi-part set
    records have already returned from classify_material_evidence().

    The role is deliberately excluded from the duplicate key. A product
    description may repeat the same composition once as a general statement
    and again after a product-type phrase such as "t-shirt made of".
    """
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, str | None, str]] = set()

    for pair in pairs:
        key = (
            float(pair["percentage"]),
            pair["material"],
            pair["raw_material_text"].casefold(),
        )
        if key not in seen:
            unique.append(pair)
            seen.add(key)

    return unique


def component_totals(pairs: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate totals separately for every explicit garment component."""
    totals: defaultdict[str, float] = defaultdict(float)
    for pair in pairs:
        totals[pair.get("role", "main")] += float(pair["percentage"])
    return {role: round(total, 2) for role, total in totals.items()}


def parse_material_text(source: str, text: str | None) -> dict[str, Any]:
    """Parse material evidence without assigning its final composition status."""
    normalised = normalise_text(text)
    percentage_pairs, ignored_non_material_percentage = extract_percentage_pairs(normalised)
    recognised_pairs = [pair for pair in percentage_pairs if pair["material_recognised"]]
    unrecognised_pairs = [pair for pair in percentage_pairs if not pair["material_recognised"]]
    missing_percentages = [
        match.group(0)
        for match in MISSING_PERCENT_RE.finditer(normalised)
        if "%" not in match.group(0)
    ]

    # A discount should not create a material-name-only claim from nearby prose.
    materials = (
        []
        if ignored_non_material_percentage and not percentage_pairs
        else find_materials(normalised)
    )
    roles = unique_preserve_order(
        extract_roles(normalised)
        + [pair["role"] for pair in percentage_pairs if pair["role"] != "main"]
    )

    return {
        "material_source": source,
        "raw_text": text or "",
        "normalised_text": normalised,
        "materials": materials,
        "primary_material": materials[0] if materials else None,
        "secondary_materials": materials[1:],
        "percentage_pairs": percentage_pairs,
        "recognised_percentage_pairs": recognised_pairs,
        "unrecognised_percentage_pairs": unrecognised_pairs,
        "percentage_total": round(sum(pair["percentage"] for pair in recognised_pairs), 2),
        "material_role_flags": roles,
        "component_totals": component_totals(recognised_pairs),
        "missing_percentage_fragments": missing_percentages,
        "contains_blend_word": "blend" in normalised,
        "ignored_non_material_percentage": ignored_non_material_percentage,
    }


def classify_material_evidence(parsed: dict[str, Any]) -> dict[str, Any]:
    """Assign one auditable composition status using conservative precedence rules."""
    result = dict(parsed)
    pairs = result.get("percentage_pairs", [])
    recognised = result.get("recognised_percentage_pairs", [])
    unrecognised = result.get("unrecognised_percentage_pairs", [])
    missing = result.get("missing_percentage_fragments", [])
    roles = result.get("material_role_flags", [])
    materials = result.get("materials", [])

    if missing or unrecognised:
        result["composition_status"] = "partial_or_invalid_percentages"
        result["percentage_total_status"] = "invalid_or_partial"
        return result

    # Structural labels such as shell, lining, cup, or trim remain separate
    # components even where only one label is present.
    #
    # A single t-shirt, legging, track pant, track jacket, bottom, or pyjama
    # label is the main garment component, not a multi-part set. A set requires
    # at least two explicitly labelled components. Tracksuit remains
    # intrinsically multi-part.
    set_component_roles = {
        "t_shirt", "legging", "pyjama", "bottom",
        "track_jacket", "track_pant", "tracksuit",
    }

    structural_roles = set(roles) - set_component_roles
    detected_set_roles = set(roles) & set_component_roles

    has_explicit_multi_part_set = (
        len(detected_set_roles) >= 2 or "tracksuit" in detected_set_roles
    )

    if pairs and (structural_roles or has_explicit_multi_part_set):
        totals = component_totals(recognised)
        valid_totals = all(99.5 <= total <= 100.5 for total in totals.values())

        result["component_totals"] = totals
        result["percentage_total"] = round(
            sum(pair["percentage"] for pair in recognised), 2
        )
        result["composition_status"] = "role_specific_or_multi_part_composition"
        result["percentage_total_status"] = (
            "multi_part" if valid_totals else "multi_part_requires_review"
        )
        return result

    calculation_pairs = unique_percentage_pairs(recognised)
    if calculation_pairs:
        total = round(sum(pair["percentage"] for pair in calculation_pairs), 2)
        ordered = sorted(calculation_pairs, key=lambda pair: pair["percentage"], reverse=True)
        primary = ordered[0]["material"]
        secondary = unique_preserve_order(
            [pair["material"] for pair in ordered[1:] if pair["material"] != primary]
        )
        result["percentage_total"] = total
        result["primary_material"] = primary
        result["secondary_materials"] = secondary

        if 99.5 <= total <= 100.5:
            result["composition_status"] = (
                "exact_single_fibre_percentage" if not secondary else "exact_blend_percentages"
            )
            result["percentage_total_status"] = "exact_100"
        else:
            result["composition_status"] = "partial_or_invalid_percentages"
            result["percentage_total_status"] = "does_not_sum_to_100"
        return result

    if result.get("contains_blend_word") and materials:
        result["composition_status"] = "named_blend_no_percentages"
        result["percentage_total_status"] = "not_available"
        return result

    if materials:
        result["composition_status"] = "material_name_only"
        result["percentage_total_status"] = "not_available"
        return result

    result["composition_status"] = "no_material_evidence"
    result["percentage_total_status"] = "not_available"
    return result