"""json_material_evidence_audit.py

Extracts and audits material evidence from product JSON files.

Important design choices:
    - Explicit percentage compositions take priority.
    - Generic Fabric dropdown values are secondary evidence.
    - Material terms are extracted only from composition-oriented text.
    - Styling text such as "wear with leather shoes" is not treated as
      garment material evidence.
    - Multi-part garments, such as bras with cup, lining, and strap
      compositions, remain in an explicit review category.

Note on structure: this module defines an original (v1) extraction/
classification implementation, then re-points `normalise_composition_text`,
`parse_material_text`, and `classify_material_evidence` to v2 adapter
functions at the bottom of the file, which wrap `material_parser_core.py`.
This is intentional v1 remains as a documented reference implementation
but it means the names used throughout `inspect_json_record()` resolve to
the v2 versions at call time. Do not remove the v1 definitions without
checking nothing else in the pipeline still relies on them directly.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import INTERIM_DATA, OUTPUTS, PRIMARY_FOLDER, RAW_DATA
from src.logging_utils import get_logger
from src.material_parser_core import (
    classify_material_evidence as core_classify_material_evidence,
    normalise_text as core_normalise_text,
    parse_material_text as core_parse_material_text,
)

logger = get_logger(__name__, log_dir=OUTPUTS)

# Paths are derived from the shared config rather than re-resolving
# PROJECT_ROOT / re-loading .env here. This avoids two files silently
# using different environment-variable names for the same directory
# (this file previously read INTERIM_DIR while config.py reads
# INTERIM_DATA_DIR they could disagree if only one was set).
CLEAN_GARMENTS_PATH = INTERIM_DATA / "garments_clean.csv"
JSON_DIR = RAW_DATA / PRIMARY_FOLDER / "styles"
INTERIM_DIR = INTERIM_DATA

OUTPUT_AUDIT_PATH = INTERIM_DIR / "json_material_evidence_audit.csv"
OUTPUT_COMPONENTS_PATH = INTERIM_DIR / "json_material_components.csv"
OUTPUT_CONFLICTS_PATH = INTERIM_DIR / "json_material_evidence_conflicts.csv"
OUTPUT_SUMMARY_PATH = INTERIM_DIR / "json_material_evidence_summary.json"


# Each key is a source phrase. Each value contains:
# canonical material category, optional material variant.
# (Deduplicated from the original "lambswool", "merino wool", and "bamboo"
# were each listed twice with identical values; harmless in a dict literal
# but removed for clarity.)
MATERIAL_ALIASES: dict[str, tuple[str, str]] = {
    "recycled polyester": ("polyester", "recycled"),
    "recycled nylon": ("nylon", "recycled"),
    "recycled polyamide": ("nylon", "recycled"),
    "recycled cotton": ("cotton", "recycled"),
    "organic cotton": ("cotton", "organic"),
    "merino wool": ("wool", "merino"),
    "lambswool": ("wool", "lambswool"),
    "polyamide": ("nylon", ""),
    "spandex": ("elastane", ""),
    "lycra": ("elastane", ""),
    "elastane": ("elastane", ""),
    "rayon": ("viscose", ""),
    "viscose": ("viscose", ""),
    "modal": ("modal", ""),
    "lyocell": ("lyocell", ""),
    "tencel": ("lyocell", "tencel"),
    "polyester": ("polyester", ""),
    "cotton": ("cotton", ""),
    "nylon": ("nylon", ""),
    "wool": ("wool", ""),
    "silk": ("silk", ""),
    "linen": ("linen", ""),
    "flax": ("linen", "flax"),
    "hemp": ("hemp", ""),
    "acrylic": ("acrylic", ""),
    "bamboo": ("bamboo", ""),
    "leather": ("leather", ""),
    "polyurethane": ("polyurethane", ""),
    "ramie": ("ramie", ""),
    "elstane": ("elastane", ""),
    "down": ("down", ""),
    "feather": ("feather", ""),
    "feathers": ("feather", ""),
    "angora": ("wool", "angora"),
    "terylene": ("polyester", "terylene"),
    "poly": ("polyester", ""),
    "elastic": ("elastane", ""),
    "elastine": ("elastane", ""),
    "spondac": ("elastane", ""),
    "mylon": ("nylon", ""),
    "polyster": ("polyester", ""),
    "elastan": ("elastane", ""),
}

ALIAS_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(MATERIAL_ALIASES, key=len, reverse=True)
)

MATERIAL_RE = re.compile(rf"\b(?P<material>{ALIAS_PATTERN})\b", flags=re.IGNORECASE)

PERCENTAGE_RE = re.compile(
    rf"(?P<percentage>\d+(?:\.\d+)?)\s*%"
    rf"\s*,?\s*"
    rf"(?:combed\s+)?"
    rf"(?P<material>{ALIAS_PATTERN})\b",
    flags=re.IGNORECASE,
)

SOURCE_PRIORITY = {
    # Explicit percentage evidence is still prioritised separately in
    # select_best_evidence().
    "materials_care_desc": 5,
    # Structured Fabric metadata is more reliable than broad narrative
    # product descriptions without percentages.
    "article_attributes_fabric": 4,
    # Description and style-note fields are lower-confidence fallbacks.
    "description": 3,
    "style_note": 2,
    # Product name is only a last-resort candidate.
    "product_display_name": 1,
}

ROLE_TERMS = {
    "body", "shell", "lining", "cup", "gusset", "strap", "back",
    "side_frame", "centre_belt", "pad", "trim", "panel", "pocket_bag",
    "upper_lining", "lower_lining",
}

COMPOSITION_MARKERS = [
    "composition:", "composition -", "composition is",
    "material and care", "materials and care", "made of", "made from",
    "fabric:", "fabric is",
]

DESCRIPTION_STOP_MARKERS = [
    " fit ", " fitting ", " wash care ", " model statistics ",
    " model's statistics ", " style note ", " pair this ", " pair them ",
    " pair with ", " wear with ", " team this ", " wear this ", " wear it ",
    " the model ",
]


def normalise_id(value: Any) -> str:
    """Convert values such as 1164.0 into the JSON filename key '1164'."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def clean_html_text(value: Any) -> str:
    """Convert raw HTML or JSON text into clean searchable text."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "na", "n/a", "none", "null"}:
        return ""
    text = html.unescape(text)
    text = text.replace("\ufeff", "").replace("ï»¿", "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>|</div>|</li>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_composition_text(value: Any) -> str:
    """Standardise composition-specific wording before extraction."""
    text = clean_html_text(value).casefold()
    if not text:
        return ""

    # Repair malformed percentage punctuation: "84%, polyamide" -> "84% polyamide".
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*,\s*", r"\1% ", text)

    # Normalise common textile blend shorthand.
    text = re.sub(r"\bpoly[\s-]*viscose\b", "polyester viscose", text)
    text = re.sub(r"\bpoly[\s-]*cotton\b", "polyester cotton", text)
    text = re.sub(r"\bpoly\s*/\s*rayon\b", "polyester viscose", text)
    text = re.sub(r"\bpoly\s*-\s*rayon\b", "polyester viscose", text)

    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def canonicalise_material(raw_material: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", raw_material.casefold()).strip()
    return MATERIAL_ALIASES[cleaned]


def get_descriptor_text(payload: dict[str, Any], descriptor_name: str) -> str:
    descriptors = payload.get("productDescriptors", {})
    if not isinstance(descriptors, dict):
        return ""
    descriptor = descriptors.get(descriptor_name, {})
    if isinstance(descriptor, dict):
        return clean_html_text(descriptor.get("value", ""))
    return clean_html_text(descriptor)


def get_fabric_attribute_text(payload: dict[str, Any]) -> str:
    """Retrieve structured Fabric, Fabric 2, Fabric 3, and Material fields."""
    attributes = payload.get("articleAttributes", {})
    if not isinstance(attributes, dict):
        return ""

    fabric_values: list[str] = []
    accepted_keys = {
        "fabric", "fabric 1", "fabric 2", "fabric 3",
        "material", "material 1", "material 2", "material 3",
    }

    for key, value in attributes.items():
        normalised_key = re.sub(r"\s+", " ", str(key).casefold()).strip()
        if normalised_key in accepted_keys:
            cleaned_value = clean_html_text(value)
            if cleaned_value:
                fabric_values.append(cleaned_value)

    return " ".join(unique_preserve_order(fabric_values))


def extract_description_composition_snippets(text: str) -> list[str]:
    """Keep only composition-relevant parts of long descriptions.

    Prevents false extraction from styling advice such as "pair with
    leather shoes".
    """
    normalised = normalise_composition_text(text)
    if not normalised:
        return []

    snippets: list[str] = []
    for marker in COMPOSITION_MARKERS:
        start_position = normalised.find(marker)
        if start_position == -1:
            continue

        snippet = normalised[start_position:start_position + 600]
        stop_positions = [
            snippet.find(stop_marker)
            for stop_marker in DESCRIPTION_STOP_MARKERS
            if snippet.find(stop_marker) > 0
        ]
        if stop_positions:
            snippet = snippet[: min(stop_positions)]
        snippets.append(snippet.strip())

    # A standalone percentage composition should still be retained even if
    # the description does not contain a recognised marker.
    if PERCENTAGE_RE.search(normalised):
        snippets.append(normalised[:600].strip())

    return unique_preserve_order(snippets)


def detect_role_headers(text: str) -> list[str]:
    """Detect role-specific composition phrases, e.g. "Shell: 97% cotton,
    3% elastane" or "Trim fabric made of 97% polyester and 3% elastane".
    """
    normalised = normalise_composition_text(text)
    if not normalised:
        return []

    roles_found: list[str] = []
    for role in ROLE_TERMS:
        role_phrase = role.replace("_", " ")
        role_pattern = rf"\b{re.escape(role_phrase)}\b(?:\s+fabric)?\s*(?::|made of)"
        if re.search(role_pattern, normalised):
            roles_found.append(role)

    return unique_preserve_order(roles_found)


def parse_material_text(source_name: str, text: str) -> dict[str, Any]:
    """Parse materials, variants, percentages, and role headers from one
    already-filtered composition text source.
    """
    normalised = normalise_composition_text(text)
    mentions: list[dict[str, str]] = []
    percentages: list[dict[str, Any]] = []

    if not normalised:
        return {
            "source": source_name, "text": "", "mentions": [],
            "percentages": [], "roles_detected": [],
        }

    for match in MATERIAL_RE.finditer(normalised):
        raw_material = match.group("material")
        canonical_material, variant = canonicalise_material(raw_material)
        mentions.append(
            {"raw_material": raw_material, "canonical_material": canonical_material, "variant": variant}
        )

    for match in PERCENTAGE_RE.finditer(normalised):
        raw_material = match.group("material")
        canonical_material, variant = canonicalise_material(raw_material)
        percentages.append(
            {
                "raw_material": raw_material,
                "canonical_material": canonical_material,
                "variant": variant,
                "percentage": float(match.group("percentage")),
                "source": source_name,
            }
        )

    deduplicated_mentions: list[dict[str, str]] = []
    seen_mentions: set[tuple[str, str]] = set()
    for mention in mentions:
        key = (mention["canonical_material"], mention["variant"])
        if key not in seen_mentions:
            deduplicated_mentions.append(mention)
            seen_mentions.add(key)

    deduplicated_percentages: list[dict[str, Any]] = []
    seen_percentages: set[tuple[str, str, float]] = set()
    for component in percentages:
        key = (component["canonical_material"], component["variant"], component["percentage"])
        if key not in seen_percentages:
            deduplicated_percentages.append(component)
            seen_percentages.add(key)

    return {
        "source": source_name,
        "text": normalised,
        "mentions": deduplicated_mentions,
        "percentages": deduplicated_percentages,
        "roles_detected": detect_role_headers(normalised),
    }


def extract_all_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract evidence from JSON fields in a controlled hierarchy."""
    materials_care_text = get_descriptor_text(payload, "materials_care_desc")
    description_text = get_descriptor_text(payload, "description")
    style_note_text = get_descriptor_text(payload, "style_note")
    product_name = clean_html_text(payload.get("productDisplayName", ""))
    fabric_attributes = get_fabric_attribute_text(payload)

    evidence_items: list[dict[str, Any]] = []

    evidence_items.append(parse_material_text("materials_care_desc", materials_care_text))

    description_snippets = extract_description_composition_snippets(description_text)
    evidence_items.append(parse_material_text("description", " ".join(description_snippets)))

    style_note_snippets = extract_description_composition_snippets(style_note_text)
    evidence_items.append(parse_material_text("style_note", " ".join(style_note_snippets)))

    evidence_items.append(parse_material_text("article_attributes_fabric", fabric_attributes))
    evidence_items.append(parse_material_text("product_display_name", product_name))

    return evidence_items


def percentage_total_status(
    percentages: list[dict[str, Any]], roles_detected: list[str]
) -> tuple[float | None, str]:
    if not percentages:
        return None, "not_available"

    total = round(sum(component["percentage"] for component in percentages), 2)

    if roles_detected and total > 100.5:
        return total, "multi_part_composition"
    if 99.5 <= total <= 100.5:
        return total, "approximately_100"
    if total < 99.5:
        return total, "incomplete_or_partial_composition"
    return total, "requires_review"


def select_best_evidence(evidence_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the strongest evidence source. Explicit percentages take
    priority over unquantified evidence.
    """
    usable = [e for e in evidence_items if e["mentions"] or e["percentages"]]
    if not usable:
        return None

    percentage_sources = [e for e in usable if e["percentages"]]
    if percentage_sources:
        return max(
            percentage_sources,
            key=lambda evidence: (SOURCE_PRIORITY[evidence["source"]], len(evidence["percentages"])),
        )

    return max(usable, key=lambda evidence: SOURCE_PRIORITY[evidence["source"]])


def classify_material_evidence(selected: dict[str, Any] | None) -> dict[str, Any]:
    """Classify material evidence for downstream scoring eligibility."""
    if selected is None:
        return {
            "primary_material": "", "secondary_materials": "", "material_variants": "",
            "composition_percentages_json": "[]", "component_count": 0,
            "percentage_total": None, "percentage_total_status": "not_available",
            "material_role_flags": "", "composition_status": "unknown",
            "material_evidence_confidence": "none",
            "scoring_eligibility": "not_eligible_missing_material_evidence",
            "exclusion_reason": "no_reliable_material_evidence",
            "elastane_present": False, "recycled_content_claim": False,
        }

    mentions = selected["mentions"]
    percentages = selected["percentages"]
    roles_detected = selected["roles_detected"]

    materials = unique_preserve_order([m["canonical_material"] for m in mentions])
    variants = unique_preserve_order(
        [f"{m['canonical_material']}:{m['variant']}" for m in mentions if m["variant"]]
    )

    total, total_status = percentage_total_status(percentages, roles_detected)

    if percentages:
        ordered_components = sorted(percentages, key=lambda c: c["percentage"], reverse=True)
        primary_material = ordered_components[0]["canonical_material"]
        secondary_materials = unique_preserve_order(
            [c["canonical_material"] for c in ordered_components[1:] if c["canonical_material"] != primary_material]
        )
    else:
        primary_material = materials[0] if materials else ""
        secondary_materials = materials[1:]

    if total_status == "multi_part_composition":
        composition_status = "role_specific_or_multi_part_composition"
        confidence = "high"
        eligibility = "needs_role_review"
        exclusion_reason = "multiple_garment_components_or_partial_percentages"

    elif percentages and total_status == "approximately_100":
        if len({c["canonical_material"] for c in percentages}) == 1:
            composition_status = "exact_single_fibre_percentage"
            confidence = "high"
            eligibility = "eligible_exact_single_fibre"
            exclusion_reason = ""
        else:
            composition_status = "exact_blend_percentages"
            confidence = "high"
            eligibility = "known_blend_pending_policy"
            exclusion_reason = "explicit_blend_requires_documented_proxy_policy"

    elif percentages:
        composition_status = "partial_or_invalid_percentages"
        confidence = "medium"
        eligibility = "needs_percentage_review"
        exclusion_reason = "percentage_total_not_suitable_for_direct_use"

    elif not percentages and len(materials) == 1 and re.search(r"\bblend\b", selected["text"]):
        primary_material = ""
        secondary_materials = materials
        composition_status = "named_blend_no_percentages"
        confidence = "medium"
        eligibility = "needs_review"
        exclusion_reason = "blend_mentioned_without_component_percentages"

    elif len(materials) == 1:
        if selected["source"] == "product_display_name":
            composition_status = "candidate_name_match"
            confidence = "low"
            eligibility = "needs_review"
            exclusion_reason = "material_inferred_from_product_name_only"
        else:
            composition_status = "single_fibre_claim_unquantified"
            confidence = "medium"
            eligibility = "main_material_proxy_only"
            exclusion_reason = "no_explicit_percentage_composition"

    elif len(materials) >= 2:
        composition_status = "named_blend_no_percentages"
        confidence = "medium"
        eligibility = "needs_review"
        exclusion_reason = "blend_without_percentages"

    else:
        composition_status = "unknown"
        confidence = "none"
        eligibility = "not_eligible_missing_material_evidence"
        exclusion_reason = "no_reliable_material_evidence"

    return {
        "primary_material": primary_material,
        "secondary_materials": "|".join(secondary_materials),
        "material_variants": "|".join(variants),
        "composition_percentages_json": json.dumps(percentages),
        "component_count": len(materials),
        "percentage_total": total,
        "percentage_total_status": total_status,
        "material_role_flags": "|".join(roles_detected),
        "composition_status": composition_status,
        "material_evidence_confidence": confidence,
        "scoring_eligibility": eligibility,
        "exclusion_reason": exclusion_reason,
        "elastane_present": "elastane" in materials,
        "recycled_content_claim": any(m["variant"] == "recycled" for m in mentions),
    }


def source_conflict_flag(evidence_items: list[dict[str, Any]]) -> tuple[bool, str]:
    """Flag meaningful disagreement between structured Fabric fields and
    detailed composition evidence.
    """
    structured_materials: set[str] = set()
    detailed_materials: set[str] = set()

    for evidence in evidence_items:
        material_set = {m["canonical_material"] for m in evidence["mentions"]}
        if evidence["source"] == "article_attributes_fabric":
            structured_materials.update(material_set)
        if evidence["source"] in {"materials_care_desc", "description", "style_note"}:
            detailed_materials.update(material_set)

    if not structured_materials or not detailed_materials:
        return False, ""

    if structured_materials.issubset(detailed_materials) or detailed_materials.issubset(structured_materials):
        return False, ""

    detail = f"structured={sorted(structured_materials)}; detailed_text={sorted(detailed_materials)}"
    return True, detail


def unwrap_product_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """JSON files contain an API wrapper with the product record in `data`."""
    nested_data = payload.get("data")
    if isinstance(nested_data, dict):
        return nested_data
    return payload


def inspect_json_record(
    garment_id: str, payload: dict[str, Any], json_path: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect one product JSON payload.

    Returns: garment-level audit record, long-format percentage component records.
    """
    payload = unwrap_product_payload(payload)

    evidence_items = extract_all_evidence(payload)
    selected = select_best_evidence(evidence_items)
    classification = classify_material_evidence(selected)

    has_conflict, conflict_detail = source_conflict_flag(evidence_items)
    json_id = normalise_id(payload.get("id"))
    selected_source = selected["source"] if selected else ""
    selected_text = selected["text"] if selected else ""

    material_sources = [e["source"] for e in evidence_items if e["mentions"]]
    extracted_material = classification["primary_material"]
    if classification["secondary_materials"]:
        extracted_material = f"{extracted_material}|{classification['secondary_materials']}"

    record = {
        "garment_id": garment_id,
        "json_file_found": True,
        "json_valid": True,
        "json_path": str(json_path) if json_path else "",
        "json_id": json_id,
        "json_id_matches_garment_id": json_id == garment_id,
        "json_error": "",
        "json_age_group": clean_html_text(payload.get("ageGroup", "")),
        "json_gender": clean_html_text(payload.get("gender", "")),
        "json_brand_name": clean_html_text(payload.get("brandName", "")),
        "json_product_display_name": clean_html_text(payload.get("productDisplayName", "")),
        "json_master_category": clean_html_text(payload.get("masterCategory", {}).get("typeName", "")),
        "json_sub_category": clean_html_text(payload.get("subCategory", {}).get("typeName", "")),
        "json_article_type": clean_html_text(payload.get("articleType", {}).get("typeName", "")),
        "json_usage": clean_html_text(payload.get("usage", "")),
        "json_season": clean_html_text(payload.get("season", "")),
        "json_base_colour": clean_html_text(payload.get("baseColour", "")),
        "raw_materials_care_desc": get_descriptor_text(payload, "materials_care_desc"),
        "raw_description_text": get_descriptor_text(payload, "description"),
        "raw_fabric_attributes": get_fabric_attribute_text(payload),
        "material_evidence_sources": "|".join(unique_preserve_order(material_sources)),
        "selected_material_evidence_source": selected_source,
        "selected_material_evidence_excerpt": selected_text[:600],
        "material_evidence": selected_text,
        "material_source": selected_source,
        "material_confidence": classification["material_evidence_confidence"],
        "material_status": classification["composition_status"],
        "extracted_material": extracted_material,
        "source_conflict": has_conflict,
        "source_conflict_detail": conflict_detail,
        **classification,
    }

    component_rows: list[dict[str, Any]] = []
    if selected:
        for component in selected["percentages"]:
            component_rows.append(
                {
                    "garment_id": garment_id,
                    "json_id": json_id,
                    "material_evidence_source": selected_source,
                    "raw_material": component["raw_material"],
                    "canonical_material": component["canonical_material"],
                    "material_variant": component["variant"],
                    "percentage": component["percentage"],
                    "component_role": component.get("role", "main"),
                    "component_totals_json": classification.get("component_totals_json", "{}"),
                    "unknown_material_terms": classification.get("unknown_material_terms", ""),
                    "malformed_percentage_expressions": classification.get(
                        "malformed_percentage_expressions", ""
                    ),
                    "material_role_flags": "|".join(selected["roles_detected"]),
                    "composition_status": classification["composition_status"],
                    "scoring_eligibility": classification["scoring_eligibility"],
                }
            )

    return record, component_rows


def build_json_index(json_dir: Path) -> dict[str, Path]:
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")

    json_index: dict[str, Path] = {}
    for path in json_dir.rglob("*.json"):
        garment_id = normalise_id(path.stem)
        if garment_id and garment_id not in json_index:
            json_index[garment_id] = path
    return json_index


def count_values(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("missing").astype(str).value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.to_dict().items()}


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAN_GARMENTS_PATH.exists():
        raise FileNotFoundError(f"Clean garments file does not exist: {CLEAN_GARMENTS_PATH}")

    garments = pd.read_csv(CLEAN_GARMENTS_PATH, low_memory=False)
    if "garment_id" not in garments.columns:
        raise KeyError("Clean garments input must contain 'garment_id'.")

    garments["garment_id"] = garments["garment_id"].apply(normalise_id)
    json_index = build_json_index(JSON_DIR)

    audit_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []

    for _, garment in garments.iterrows():
        garment_id = garment["garment_id"]
        json_path = json_index.get(garment_id)

        base_record = {
            "garment_id": garment_id,
            "csv_gender": garment.get("gender", ""),
            "csv_master_category": garment.get("master_category", ""),
            "csv_sub_category": garment.get("sub_category", ""),
            "csv_article_type": garment.get("article_type", ""),
            "csv_product_name": garment.get("product_name", ""),
        }

        if json_path is None:
            audit_records.append(
                {
                    **base_record,
                    "json_file_found": False, "json_valid": False, "json_path": "",
                    "json_id": "", "json_id_matches_garment_id": False,
                    "json_error": "json_file_not_found", "extracted_material": "",
                    "primary_material": "", "secondary_materials": "",
                    "composition_status": "unknown", "material_evidence_confidence": "none",
                    "scoring_eligibility": "not_eligible_missing_material_evidence",
                    "exclusion_reason": "json_file_not_found", "source_conflict": False,
                }
            )
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            record, components = inspect_json_record(
                garment_id=garment_id, payload=payload, json_path=json_path,
            )
            audit_records.append({**base_record, **record})
            component_records.extend(components)

        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
            audit_records.append(
                {
                    **base_record,
                    "json_file_found": True, "json_valid": False,
                    "json_path": str(json_path), "json_id": "",
                    "json_id_matches_garment_id": False,
                    "json_error": type(error).__name__, "extracted_material": "",
                    "primary_material": "", "secondary_materials": "",
                    "composition_status": "unknown", "material_evidence_confidence": "none",
                    "scoring_eligibility": "not_eligible_missing_material_evidence",
                    "exclusion_reason": "invalid_or_unreadable_json", "source_conflict": False,
                }
            )

    audit_df = pd.DataFrame(audit_records)
    components_df = pd.DataFrame(component_records)

    audit_df.to_csv(OUTPUT_AUDIT_PATH, index=False)

    if components_df.empty:
        components_df = pd.DataFrame(
            columns=[
                "garment_id", "json_id", "material_evidence_source", "raw_material",
                "canonical_material", "material_variant", "percentage", "component_role",
                "component_totals_json", "unknown_material_terms",
                "malformed_percentage_expressions", "material_role_flags",
                "composition_status", "scoring_eligibility",
            ]
        )
    components_df.to_csv(OUTPUT_COMPONENTS_PATH, index=False)

    conflicts_df = audit_df.loc[audit_df["source_conflict"].fillna(False)].copy()
    conflicts_df.to_csv(OUTPUT_CONFLICTS_PATH, index=False)

    summary = {
        "input_garments": int(len(audit_df)),
        "json_files_found": int(audit_df["json_file_found"].sum()),
        "valid_json_files": int(audit_df["json_valid"].sum()),
        "missing_json_files": int((~audit_df["json_file_found"]).sum()),
        "json_id_mismatches": int(
            (
                audit_df["json_file_found"]
                & audit_df["json_valid"]
                & (~audit_df["json_id_matches_garment_id"])
            ).sum()
        ),
        "source_conflicts": int(audit_df["source_conflict"].fillna(False).sum()),
        "composition_status_counts": count_values(audit_df["composition_status"]),
        "scoring_eligibility_counts": count_values(audit_df["scoring_eligibility"]),
        "primary_material_counts": count_values(audit_df["primary_material"]),
        "output_files": {
            "audit": str(OUTPUT_AUDIT_PATH),
            "components": str(OUTPUT_COMPONENTS_PATH),
            "conflicts": str(OUTPUT_CONFLICTS_PATH),
            "summary": str(OUTPUT_SUMMARY_PATH),
        },
    }

    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logger.info("JSON material evidence audit completed.")
    logger.info("Input garments: %s", f"{len(audit_df):,}")
    logger.info("Valid JSON files: %s", f"{summary['valid_json_files']:,}")
    logger.info("Missing JSON files: %s", f"{summary['missing_json_files']:,}")
    logger.info("JSON ID mismatches: %s", f"{summary['json_id_mismatches']:,}")
    logger.info("Source conflicts: %s", f"{summary['source_conflicts']:,}")
    logger.info("Audit file: %s", OUTPUT_AUDIT_PATH)
    logger.info("Components file: %s", OUTPUT_COMPONENTS_PATH)
    logger.info("Conflicts file: %s", OUTPUT_CONFLICTS_PATH)
    logger.info("Summary file: %s", OUTPUT_SUMMARY_PATH)



# Parser v2 integration layer


# Preserve the original text cleaner because it repairs project-specific
# shorthand such as "poly viscose" before parser v2 receives the text.
normalise_composition_text_legacy = normalise_composition_text


def normalise_composition_text_v2(value: Any) -> str:
    """Apply the original project cleaning, then parser-v2 punctuation cleaning."""
    legacy_cleaned = normalise_composition_text_legacy(value)
    return core_normalise_text(legacy_cleaned)


def material_variant_from_fragment(
    raw_fragment: str, fallback_material: str
) -> tuple[str, str]:
    """Recover the project-specific material variant where possible.

    Examples: organic cotton -> cotton, organic; recycled polyester ->
    polyester, recycled; spandex -> elastane, blank variant.
    """
    match = MATERIAL_RE.search(raw_fragment)
    if match:
        raw_material = match.group("material")
        return canonicalise_material(raw_material)
    return fallback_material, ""


def parse_material_text_v2(source_name: str, text: str) -> dict[str, Any]:
    """Adapt parser v2 output to the existing audit-pipeline structure.

    The rest of the pipeline still receives: mentions, percentages,
    roles_detected, source, text.
    """
    normalised = normalise_composition_text_v2(text)
    core_result = core_parse_material_text(source_name, normalised)

    mentions: list[dict[str, str]] = []
    seen_mentions: set[tuple[str, str]] = set()

    for match in MATERIAL_RE.finditer(normalised):
        raw_material = match.group("material")
        canonical_material, variant = canonicalise_material(raw_material)
        key = (canonical_material, variant)
        if key not in seen_mentions:
            mentions.append(
                {"raw_material": raw_material, "canonical_material": canonical_material, "variant": variant}
            )
            seen_mentions.add(key)

    # Add parser-v2 aliases not covered by the older alias dictionary,
    # e.g. "elasthen".
    for canonical_material in core_result["materials"]:
        if not any(m["canonical_material"] == canonical_material for m in mentions):
            mentions.append(
                {"raw_material": canonical_material, "canonical_material": canonical_material, "variant": ""}
            )

    percentages: list[dict[str, Any]] = []
    seen_percentages: set[tuple[str, str, float]] = set()

    for pair in core_result["recognised_percentage_pairs"]:
        raw_fragment = pair["raw_material_text"]
        canonical_material, variant = material_variant_from_fragment(raw_fragment, pair["material"])
        percentage = float(pair["percentage"])
        key = (canonical_material, variant, percentage)
        if key not in seen_percentages:
            percentages.append(
                {
                    "raw_material": raw_fragment, "canonical_material": canonical_material,
                    "variant": variant, "percentage": percentage, "source": source_name,
                }
            )
            seen_percentages.add(key)

    legacy_roles = detect_role_headers(normalised)
    roles_detected = unique_preserve_order(core_result["material_role_flags"] + legacy_roles)

    return {
        "source": source_name, "text": normalised, "mentions": mentions,
        "percentages": percentages, "roles_detected": roles_detected,
        "core_result": core_result,
    }


def classify_material_evidence_v2(selected: dict[str, Any] | None) -> dict[str, Any]:
    """Use parser-v2 classification while preserving the output columns
    required by 03b_extract_full_json_materials.py and later scripts.
    """
    if selected is None:
        return {
            "primary_material": "", "secondary_materials": "", "material_variants": "",
            "composition_percentages_json": "[]", "component_count": 0,
            "percentage_total": None, "percentage_total_status": "not_available",
            "material_role_flags": "", "composition_status": "unknown",
            "material_evidence_confidence": "none",
            "scoring_eligibility": "not_eligible_missing_material_evidence",
            "exclusion_reason": "no_reliable_material_evidence",
            "elastane_present": False, "recycled_content_claim": False,
            "unknown_material_terms": "", "malformed_percentage_expressions": "",
            "component_totals_json": "{}",
        }

    core_result = selected["core_result"]
    core_classification = core_classify_material_evidence(core_result)

    mentions = selected["mentions"]
    percentages = selected["percentages"]
    roles_detected = selected["roles_detected"]

    materials = unique_preserve_order([m["canonical_material"] for m in mentions])
    variants = unique_preserve_order(
        [f"{m['canonical_material']}:{m['variant']}" for m in mentions if m["variant"]]
    )

    status = core_classification["composition_status"]
    total = core_classification["percentage_total"]

    unknown_material_terms = "|".join(
        pair["raw_material_text"] for pair in core_result["unrecognised_percentage_pairs"]
    )
    malformed_percentage_expressions = "|".join(core_result["missing_percentage_fragments"])

    if percentages:
        ordered_components = sorted(percentages, key=lambda c: c["percentage"], reverse=True)
        primary_material = ordered_components[0]["canonical_material"]
        secondary_material_list = unique_preserve_order(
            [c["canonical_material"] for c in ordered_components[1:] if c["canonical_material"] != primary_material]
        )
    else:
        primary_material = materials[0] if materials else ""
        secondary_material_list = materials[1:]

    if status == "role_specific_or_multi_part_composition":
        confidence = "high"
        eligibility = "needs_role_review"
        exclusion_reason = "multiple_garment_components_or_partial_percentages"
        total_status = "multi_part_composition"

    elif status == "exact_single_fibre_percentage":
        confidence = "high"
        eligibility = "eligible_exact_single_fibre"
        exclusion_reason = ""
        total_status = "approximately_100"

    elif status == "exact_blend_percentages":
        component_materials = {c["canonical_material"] for c in percentages}
        confidence = "high"
        total_status = "approximately_100"

        if len(component_materials) == 1:
            status = "exact_single_fibre_percentage"
            eligibility = "eligible_exact_single_fibre"
            exclusion_reason = ""
        else:
            eligibility = "known_blend_pending_policy"
            exclusion_reason = "explicit_blend_requires_documented_proxy_policy"

    elif status == "partial_or_invalid_percentages":
        confidence = "medium"
        eligibility = "needs_percentage_review"
        exclusion_reason = "percentage_total_not_suitable_for_direct_use"
        total_status = "incomplete_or_partial_composition" if (total is not None and total < 99.5) else "requires_review"

    elif status == "named_blend_no_percentages":
        # Do not treat one ingredient of an unquantified blend as primary.
        primary_material = ""
        secondary_material_list = materials
        confidence = "medium"
        eligibility = "needs_review"
        exclusion_reason = "blend_without_percentages"
        total_status = "not_available"

    elif status == "material_name_only" and len(materials) >= 2:
        primary_material = ""
        secondary_material_list = materials
        confidence = "medium"
        eligibility = "needs_review"
        exclusion_reason = "multiple_materials_without_percentages"
        total_status = "not_available"
        status = "named_blend_no_percentages"

    elif status == "material_name_only":
        confidence = "medium"
        total_status = "not_available"

        if selected["source"] == "product_display_name":
            status = "candidate_name_match"
            confidence = "low"
            eligibility = "needs_review"
            exclusion_reason = "material_inferred_from_product_name_only"
        else:
            status = "single_fibre_claim_unquantified"
            eligibility = "main_material_proxy_only"
            exclusion_reason = "no_explicit_percentage_composition"

    else:
        status = "unknown"
        primary_material = ""
        secondary_material_list = []
        confidence = "none"
        eligibility = "not_eligible_missing_material_evidence"
        exclusion_reason = "no_reliable_material_evidence"
        total_status = "not_available"

    return {
        "primary_material": primary_material,
        "secondary_materials": "|".join(secondary_material_list),
        "material_variants": "|".join(variants),
        "composition_percentages_json": json.dumps(percentages),
        "component_count": len(materials),
        "percentage_total": total,
        "percentage_total_status": total_status,
        "material_role_flags": "|".join(roles_detected),
        "composition_status": status,
        "material_evidence_confidence": confidence,
        "scoring_eligibility": eligibility,
        "exclusion_reason": exclusion_reason,
        "elastane_present": "elastane" in materials,
        "recycled_content_claim": any(m["variant"] == "recycled" for m in mentions),
        "unknown_material_terms": unknown_material_terms,
        "malformed_percentage_expressions": malformed_percentage_expressions,
        "component_totals_json": "{}",
    }


# Use parser v2 while retaining the established JSON evidence hierarchy,
# source priority, output columns, and pipeline structure.
normalise_composition_text = normalise_composition_text_v2
parse_material_text = parse_material_text_v2
classify_material_evidence = classify_material_evidence_v2


if __name__ == "__main__":
    main()