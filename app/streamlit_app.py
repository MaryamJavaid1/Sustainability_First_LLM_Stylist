"""streamlit_app.py

Web interface for the ReWear sustainability-first wardrobe stylist.

This is a supplementary, non-evaluative research demonstrator (see Section
4.13 of the dissertation): it presents the same rule-based selection engine
used to construct the fine-tuning dataset, with an optional Gemini-authored
explanation layer. It is not the fine-tuned model evaluated in Chapter 5.
"""

from __future__ import annotations

import base64
import html
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from app.recommender import detect_garment_type, ollama_available, recommend

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "wardrobe_scored.csv"
IMAGE_DIR = PROJECT_ROOT / "app" / "assets" / "garments"
STYLE_PATH = Path(__file__).resolve().parent / "style.css"

# Occasion configuration

EVENTS = [
    "casual lunch",
    "office meeting",
    "university class",
    "formal dinner",
    "weekend errand",
]

EVENT_LABELS = {
    "casual lunch": "Casual lunch",
    "office meeting": "Office meeting",
    "university class": "University class",
    "formal dinner": "Formal dinner",
    "weekend errand": "Weekend errand",
}

# NOTE: this is a single indicative season shown to the user before a
# recommendation is generated (see the "will first consider..." caption in
# the "Create a look" tab). recommender.py's EVENT_SEASON_MAP holds the full
# ordered season-preference list actually used during selection this dict
# only surfaces the first entry as a pre-generation hint and is not used to
# label an actual result (results use OutfitResult.matched_season instead).
EVENT_SEASONS = {
    "casual lunch": "Summer",
    "office meeting": "Fall",
    "university class": "Fall",
    "formal dinner": "Winter",
    "weekend errand": "Summer",
}

# Small formatting / rendering utilities

def render_html(template: str) -> None:
    """Render an HTML template via st.markdown, safely.

    Dedents the template (so multi-line f-strings written at any indentation
    level produce clean HTML) and strips blank lines. A blank line in the
    middle of an HTML fragment causes Streamlit's underlying CommonMark
    parser to terminate the HTML block early, leaving trailing tags to be
    rendered as literal text stripping blank lines avoids this class of
    rendering bug entirely.
    """
    cleaned = textwrap.dedent(template).strip()
    cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip() != "")
    st.markdown(cleaned, unsafe_allow_html=True)


def render_mode_note(text: str) -> None:
    """Render a single-line disclosure/limitation note using the shared
    `.rw-mode-note` style (e.g. "this is a rule-based comparison, not two
    fine-tuned models" or a fallback-relaxation disclosure)."""
    st.markdown(f"<div class='rw-mode-note'>{text}</div>", unsafe_allow_html=True)


def score_colour(score: float) -> str:
    """Map a 1-10 sustainability proxy score to an accessible earthy colour
    for the score badge on a garment card."""
    if score >= 7:
        return "#2f7650"
    if score >= 5.5:
        return "#6f9874"
    if score >= 4:
        return "#b08b38"
    if score >= 2.5:
        return "#b46c43"
    return "#a65045"


def clean_text(value: Any, fallback: str = "Not available") -> str:
    """Coerce a possibly-missing/NaN dataset value to a display string,
    substituting `fallback` for None, NaN, or an empty/whitespace-only value."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def owner_label(owner: Any) -> str:
    """Human-facing wardrobe label for a raw wardrobe_owner_id value, e.g.
    "1" -> "Wardrobe 01". Owner IDs in the dataset are already 1-indexed."""
    raw = clean_text(owner, "0")
    try:
        return f"Wardrobe {int(float(raw)):02d}"
    except (TypeError, ValueError):
        return f"Wardrobe {raw}"


def _owner_sort_key(value: str) -> tuple[int, Any]:
    """Sort key placing owner IDs in true numeric order (1, 2, ..., 10, 11)
    rather than Python's default lexicographic string order (1, 10, 11, ...,
    2). Non-numeric values sort after all numeric ones."""
    try:
        return (0, int(float(value)))
    except (TypeError, ValueError):
        return (1, value)


def build_owner_options(wardrobe: pd.DataFrame) -> dict[str, str]:
    """Build a {display label -> owner_key} mapping for a wardrobe-picker
    selectbox, with owners listed in true numeric order."""
    owners = sorted(wardrobe["owner_key"].dropna().unique().tolist(), key=_owner_sort_key)
    return {owner_label(owner): owner for owner in owners}


def detect_event(question: str) -> str | None:
    """Match free text to one of the known occasion contexts by keyword,
    returning None if no occasion keyword is present."""
    query = question.casefold()
    keyword_map = {
        "casual lunch": ["lunch", "brunch", "cafe"],
        "office meeting": ["office", "meeting", "work", "interview"],
        "university class": ["class", "university", "college", "campus", "lecture"],
        "formal dinner": ["formal", "dinner", "evening event", "gala"],
        "weekend errand": ["errand", "weekend", "shopping", "running around"],
    }
    for event, keywords in keyword_map.items():
        if any(keyword in query for keyword in keywords):
            return event
    return None


def get_impact_value(item: dict[str, Any]) -> float | None:
    """Resolve a garment's CO2e production-impact proxy from whichever
    field is present, checking the wardrobe-explorer-assigned "co2e_kg" key
    first, then known dataset column names in order of recency. Returns
    None (never a fabricated number) if no field holds a usable value."""
    for column in ("co2e_kg", "reuse_avoided_co2e_kg", "estimated_production_co2e_kg"):
        value = item.get(column)
        try:
            if value is not None and pd.notna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None

# Data loading

@st.cache_data(show_spinner=False)
def load_wardrobe() -> pd.DataFrame:
    """Load the scored wardrobe dataset, filling any expected-but-missing
    columns with an empty placeholder and coercing numeric columns so
    downstream filtering/sorting never fails on a malformed value. Returns
    an empty DataFrame if the data file has not been generated yet."""
    if not DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE, low_memory=False)
    required_columns = [
        "garment_id",
        "product_name",
        "sub_category",
        "season",
        "reuse_count",
        "sustainability_score",
        "extracted_material",
        "primary_material",
        "wardrobe_owner_id",
        "base_colour",
        "wearability_label",
        "second_hand_status",
        "gender",
        "usage",
        "reuse_avoided_co2e_kg",
        "estimated_production_co2e_kg",
    ]
    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    for column in ("reuse_count", "sustainability_score"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["garment_id"] = df["garment_id"].astype(str).str.replace(r"\\.0$", "", regex=True)
    df["owner_key"] = df["wardrobe_owner_id"].astype(str)
    return df


def image_data_uri(garment_id: str) -> str | None:
    """Base64-encode a garment's image file as a data URI for inline
    embedding, or None if no image exists on disk for that garment."""
    image_path = IMAGE_DIR / f"{garment_id}.jpg"
    if not image_path.exists():
        return None
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except OSError:
        return None


def load_app_css() -> str:
    """Load the app stylesheet from style.css and wrap it for injection via
    st.markdown(unsafe_allow_html=True). Returns an empty (no-op) <style>
    block if the file is missing, so the app still runs unstyled rather
    than broken if style.css is ever absent from a deployment."""
    try:
        css_text = STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        st.warning(f"Stylesheet not found at {STYLE_PATH.name} continuing with default styling.")
        return "<style></style>"
    return f"<style>\n{css_text}\n</style>"

# Garment / recommendation rendering

def render_garment_card(item: dict[str, Any], column: Any) -> None:
    """Render a single garment as a card (image, name, category, wear
    count, material, sustainability score, and CO2e proxy) inside the given
    Streamlit column."""
    garment_id = clean_text(item.get("garment_id"), "").replace(".0", "")
    image = image_data_uri(garment_id)
    score = float(item.get("sustainability_score", 0) or 0)
    impact = get_impact_value(item)

    name = html.escape(clean_text(item.get("product_name"), "Unnamed garment"))
    category = html.escape(clean_text(item.get("sub_category")))
    material = html.escape(clean_text(item.get("material", item.get("extracted_material"))))
    reuse_count = int(float(item.get("reuse_count", 0) or 0))

    if image:
        visual = f'<img class="rw-card-image" src="{image}" alt="{name}">'
    else:
        visual = '<div class="rw-image-fallback">Garment image unavailable</div>'

    if impact is not None:
        impact_line = (
            f"<div class='rw-card-meta' style='margin-top:0.55rem;'>"
            f"Indicative production proxy: {impact:.1f} kg CO2e</div>"
        )
    else:
        impact_line = (
            "<div class='rw-card-meta' style='margin-top:0.55rem;'>"
            "Production proxy: not available for this material</div>"
        )

    with column:
        render_html(f"""
            <div class="rw-card">
                {visual}
                <div class="rw-card-body">
                    <div class="rw-card-title">{name}</div>
                    <div class="rw-card-meta">{category} \u00b7 worn {reuse_count} times</div>
                    <span class="rw-chip">{material}</span>
                    <span class="rw-score" style="background:{score_colour(score)};">Proxy score {score:.1f}/10</span>
                    {impact_line}
                </div>
            </div>
        """)


def render_evidence(result: Any, event: str) -> None:
    """Render the "Recommendation evidence" panel: wardrobe source,
    occasion/season context, material evidence, reuse totals, and the
    aggregate CO2e proxy for a given OutfitResult."""
    impacts = [get_impact_value(item) for item in result.items]
    usable_impacts = [value for value in impacts if value is not None]
    total_impact = sum(usable_impacts) if usable_impacts else None
    chosen_materials = ", ".join(
        sorted({clean_text(item.get("material"), "Unspecified") for item in result.items})
    )
    completeness = (
        "Complete outfit selected" if len(result.items) >= 2 else "Closest available wardrobe option"
    )
    impact_value = f"{total_impact:.1f} kg CO2e" if total_impact is not None else "Not available"

    # Reflects the season the recommendation actually matched, not a static
    # per-occasion default see OutfitResult.matched_season.
    if result.source == "fine_tuned_local":
        season_display = "Not tracked in this mode"
    else:
        season_display = result.matched_season if result.matched_season else "Not season-matched (see note above)"
    render_html(f"""
        <div class="rw-evidence">
            <div class="rw-evidence-title">Recommendation evidence</div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Wardrobe source</span>
                <span class="rw-evidence-value">Existing garments only</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Occasion context</span>
                <span class="rw-evidence-value">{html.escape(EVENT_LABELS[event])}</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Season context</span>
                <span class="rw-evidence-value">{html.escape(season_display)}</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Material evidence</span>
                <span class="rw-evidence-value">{html.escape(chosen_materials)}</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Combined prior wears</span>
                <span class="rw-evidence-value">{result.total_reuse}</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Average proxy score</span>
                <span class="rw-evidence-value">{result.avg_sustainability:.2f}/10</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Outfit status</span>
                <span class="rw-evidence-value">{completeness}</span>
            </div>
            <div class="rw-evidence-row">
                <span class="rw-evidence-label">Equivalent new-garment production proxy</span>
                <span class="rw-evidence-value">{impact_value}</span>
            </div>
        </div>
    """)
    st.caption(
        "Production-impact values are indicative decision-support proxies for an equivalent new garment. "
        "They are not measured emissions savings."
    )


def render_result(
    result: Any,
    warning: str | None,
    event: str,
    heading: str,
    wardrobe: pd.DataFrame | None = None,
) -> None:
    """Render a full recommendation result: heading, any LLM-fallback
    warning, the rationale box, any constraint-relaxation disclosure, and
    the garment cards with their evidence panel.

    If the result has no items (e.g. a specifically requested garment type
    was not found in this wardrobe), renders the rationale as a warning and,
    when `wardrobe` is supplied, lists the article types actually available
    so the user knows what to ask for instead.
    """
    if warning:
        st.warning(warning)

    source_labels = {
        "fine_tuned_local": "Fine-tuned model (local)",
    }
    source_label = source_labels.get(result.source, "Rule-based ReWear engine")
    source_label = source_labels.get(result.source, "Rule-based ReWear engine")
    rationale = html.escape(clean_text(result.rationale, "No explanation was generated.")).replace("\n", "<br>")

    st.markdown(f"## {heading}")

    if not result.items:
        st.warning(rationale)
        if wardrobe is not None and not wardrobe.empty and "article_type" in wardrobe.columns:
            available = sorted(wardrobe["article_type"].dropna().unique().tolist())
            if available:
                st.caption("This wardrobe currently has: " + ", ".join(available))
        return

    render_html(f"""
        <div class="rw-look">
            <div class="rw-source">{source_label}</div>
            <div class="rw-rationale">{rationale}</div>
        </div>
    """)

    if result.fallback_note:
        render_mode_note(html.escape(result.fallback_note))

    left, right = st.columns([2.15, 1], gap="large")
    with left:
        cards = st.columns(len(result.items))
        for item, column in zip(result.items, cards):
            render_garment_card(item, column)
    with right:
        render_evidence(result, event)


def render_comparison(wardrobe: pd.DataFrame, event: str) -> None:
    """Render sustainability-first and least-worn recommendations side by
    side, both rule-based, for direct visual comparison."""
    sustainability_result, sustainability_warning = recommend(
        wardrobe, event, style_first=False
    )
    style_result, style_warning = recommend(wardrobe, event, style_first=True)
    st.markdown("## Compare selection strategies")
    render_mode_note(
        "This comparison demonstrates two rule-based selection strategies. "
        "It does not compare separately fine-tuned language models."
    )

    st.markdown("### Sustainability-first")
    st.caption("Ranks eligible garments by the environmental and circularity proxy score.")
    render_result(sustainability_result, sustainability_warning, event, "")

    st.divider()

    st.markdown("### Least-worn comparison")
    st.caption("Ranks eligible garments by lowest prior wear count. This is a rule-based comparison mode.")
    render_result(style_result, style_warning, event, "")


def render_wardrobe_explorer(wardrobe: pd.DataFrame) -> None:
    """Render the "Explore wardrobe" tab: filterable, sortable browsing of
    every scored garment across the whole dataset (not scoped to one
    wardrobe owner)."""
    st.markdown("## Wardrobe explorer")
    st.caption("Browse scored garments and inspect the available evidence used by the prototype.")

    search_col, season_col, category_col, score_col = st.columns([1.4, 1, 1.2, 1])
    search_query = search_col.text_input("Search garments", placeholder="Search by item name")
    seasons = ["All seasons"] + sorted(
        value for value in wardrobe["season"].dropna().astype(str).unique() if value
    )
    season_filter = season_col.selectbox("Season", seasons)
    categories = ["All categories"] + sorted(
        value for value in wardrobe["sub_category"].dropna().astype(str).unique() if value
    )
    category_filter = category_col.selectbox("Category", categories)
    minimum_score = score_col.slider("Minimum proxy score", 1.0, 10.0, 1.0, 0.5)

    material_col, wear_col, sort_col = st.columns([1.2, 1.2, 1.4])
    materials = ["All materials"] + sorted(
        value for value in wardrobe["extracted_material"].dropna().astype(str).unique() if value
    )
    material_filter = material_col.selectbox("Material", materials)
    wearability_values = ["All wearability levels"] + sorted(
        value for value in wardrobe["wearability_label"].dropna().astype(str).unique() if value
    )
    wearability_filter = wear_col.selectbox("Wearability", wearability_values)
    sort_filter = sort_col.selectbox(
        "Sort by", ["Highest proxy score", "Most worn", "Least worn", "Material"]
    )

    filtered = wardrobe.copy()
    if search_query.strip():
        filtered = filtered[
            filtered["product_name"].astype(str).str.contains(search_query.strip(), case=False, na=False)
        ]
    if season_filter != "All seasons":
        filtered = filtered[filtered["season"].astype(str) == season_filter]
    if category_filter != "All categories":
        filtered = filtered[filtered["sub_category"].astype(str) == category_filter]
    if material_filter != "All materials":
        filtered = filtered[filtered["extracted_material"].astype(str) == material_filter]
    if wearability_filter != "All wearability levels":
        filtered = filtered[filtered["wearability_label"].astype(str) == wearability_filter]
    filtered = filtered[filtered["sustainability_score"] >= minimum_score]

    if sort_filter == "Highest proxy score":
        filtered = filtered.sort_values("sustainability_score", ascending=False)
    elif sort_filter == "Most worn":
        filtered = filtered.sort_values("reuse_count", ascending=False)
    elif sort_filter == "Least worn":
        filtered = filtered.sort_values("reuse_count", ascending=True)
    else:
        filtered = filtered.sort_values("extracted_material", ascending=True)

    st.caption(f"Showing {min(len(filtered), 24):,} of {len(filtered):,} matching garments")
    records = filtered.head(24).to_dict("records")

    for start in range(0, len(records), 3):
        cards = st.columns(3)
        for item, column in zip(records[start : start + 3], cards):
            item["material"] = item.get("extracted_material") or item.get("primary_material")
            item["co2e_kg"] = item.get(
                "counterfactual_new_garment_production_co2e_kg", item.get("estimated_production_co2e_kg")
            )
            render_garment_card(item, column)

def render_sidebar(wardrobe: pd.DataFrame) -> bool:
    """Render the workspace sidebar (dataset stats, engine toggle,
    transparency note). Returns use_fine_tuned."""
    with st.sidebar:
        st.markdown("## Workspace")
        st.caption("Research demonstrator")
        st.metric("Scored garments", f"{len(wardrobe):,}")
        st.metric("Synthetic wardrobes", f"{wardrobe['owner_key'].nunique():,}")
        st.divider()

        ollama_ready = ollama_available()
        use_fine_tuned = False
        if ollama_ready:
            use_fine_tuned = st.toggle(
                "Use fine-tuned model (local, experimental)", value=False
            )
            if use_fine_tuned:
                st.caption("Using the fine-tuned Llama-3-8B model, running locally.")
            else:
                st.caption("Rule-based recommendations remain fully available either way.")
        else:
            st.caption("Fine-tuned local model unavailable (Ollama not running).")
            st.caption("Rule-based recommendations remain fully available.")

        st.divider()
        st.markdown("### Transparency")
        st.caption(
            "ReWear ranks garments using available material metadata and synthetic reuse, "
            "wearability, and second-hand-status variables. It does not calculate a full product life-cycle assessment."
        )

    return use_fine_tuned
# Page sections

def render_create_a_look_tab(wardrobe: pd.DataFrame, use_fine_tuned: bool) -> None:
    """Render the "Create a look" tab: wardrobe/occasion/strategy selection,
    the generate button, and (once generated) the resulting recommendation
    or side-by-side comparison."""
    render_html(f"""
        <section class="rw-hero">
            <div class="rw-eyebrow">Style from what you already own</div>
            <h1>A considered look, built from your wardrobe.</h1>
            <p>ReWear selects suitable garments from an existing wardrobe using occasion, season,
            material evidence, reuse history, wearability, and a transparent environmental and circularity proxy.</p>
        </section>
    """)

    owner_options = build_owner_options(wardrobe)

    with st.container(border=True):
        st.markdown("### <span class='rw-step'>1</span>Select a wardrobe", unsafe_allow_html=True)
        selected_label = st.selectbox("Wardrobe", list(owner_options.keys()), label_visibility="collapsed")
        selected_owner = owner_options[selected_label]
        selected_wardrobe = wardrobe[wardrobe["owner_key"] == selected_owner]

        summary_a, summary_b, summary_c = st.columns(3)
        summary_a.metric("Garments", f"{len(selected_wardrobe):,}")
        summary_b.metric("Average proxy score", f"{selected_wardrobe['sustainability_score'].mean():.2f}/10")
        summary_c.metric("Categories", selected_wardrobe["sub_category"].nunique())

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### <span class='rw-step'>2</span>Choose an occasion", unsafe_allow_html=True)
        selected_event = st.radio(
            "Occasion",
            EVENTS,
            horizontal=True,
            format_func=lambda event: EVENT_LABELS[event],
            label_visibility="collapsed",
        )
        st.caption(
            f"ReWear will first consider garments suitable for {EVENT_LABELS[selected_event].lower()} "
            f"and the {EVENT_SEASONS[selected_event]} context."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### <span class='rw-step'>3</span>Choose a selection view", unsafe_allow_html=True)
        selection_view = st.radio(
            "Selection view",
            ["Sustainability-first", "Compare both", "Least-worn comparison"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if selection_view == "Least-worn comparison":
            render_mode_note(
                "Least-worn comparison ranks suitable garments by lowest prior wear count. "
                "It is a rule-based comparison strategy, not a separately fine-tuned model."
            )
        elif selection_view == "Compare both":
            render_mode_note(
                "Compare both displays sustainability-first and least-worn rule-based selections side by side. "
                "This is not a comparison between separately fine-tuned language models."
            )

        generate = st.button("Create my look", type="primary", use_container_width=True)

    if generate:
        with st.spinner("Curating a look from the selected wardrobe..."):
            if selection_view == "Compare both":
                st.session_state["comparison_request"] = (selected_owner, selected_event)
                st.session_state.pop("single_request", None)
            else:
                style_first = selection_view == "Least-worn comparison"
                result, warning = recommend(
                    selected_wardrobe,
                    selected_event,
                    style_first=style_first,
                    use_fine_tuned=use_fine_tuned and not style_first,
                )
                st.session_state["single_request"] = (
                    result,
                    warning,
                    selected_event,
                    "Your ReWear look" if not style_first else "Your least-worn comparison look",
                )
                st.session_state.pop("comparison_request", None)

    if "single_request" in st.session_state:
        result, warning, event, heading = st.session_state["single_request"]
        render_result(result, warning, event, heading)
    elif "comparison_request" in st.session_state:
        owner, event = st.session_state["comparison_request"]
        render_comparison(wardrobe[wardrobe["owner_key"] == owner], event)


def render_ask_rewear_tab(wardrobe: pd.DataFrame, use_fine_tuned: bool) -> None:
    """Render the "Ask ReWear" tab: free-text request entry with detected
    garment-type/occasion feedback, and the resulting recommendation."""
    st.markdown("## Ask ReWear in your own words")
    st.caption("ReWear maps your request to one of the available occasion contexts before selecting from the chosen wardrobe.")

    ask_left, ask_right = st.columns([1, 1.5], gap="large")
    with ask_left:
        owner_labels = build_owner_options(wardrobe)
        question_owner_label = st.selectbox("Wardrobe", list(owner_labels.keys()), key="question_owner")
        question_owner = owner_labels[question_owner_label]
        question = st.text_area(
            "What would you like to wear?",
            placeholder="For example: I have a job interview tomorrow. What should I wear?",
            height=130,
        )

        detected_event = detect_event(question) if question.strip() else None
        detected_garment = detect_garment_type(question) if question.strip() else None

        if detected_garment:
            st.success(f"Detected garment request: {detected_garment}")
        elif question.strip():
            st.caption("No specific garment type detected \u2014 ReWear will suggest a complete outfit instead.")

        if detected_event:
            st.success(f"Detected context: {EVENT_LABELS[detected_event]}")
            selected_question_event = detected_event
        else:
            selected_question_event = st.selectbox(
                "Choose an occasion context", EVENTS, format_func=lambda event: EVENT_LABELS[event]
            )
        ask_button = st.button("Ask ReWear", type="primary", use_container_width=True)

    with ask_right:
        st.markdown("### Example requests")
        st.markdown("- I have an office meeting tomorrow. What should I wear?")
        st.markdown("- Help me choose something for a casual lunch.")
        st.markdown("- What can I wear for a formal dinner?")
        st.markdown(
            "<div class='rw-prototype-note'>The recommendation engine selects garments using the available "
            "wardrobe data. Optional Gemini support, when enabled, provides explanation wording only.</div>",
            unsafe_allow_html=True,
        )

    if ask_button:
        if not question.strip():
            st.warning("Enter a request before asking ReWear for a recommendation.")
            return
        selected_wardrobe = wardrobe[wardrobe["owner_key"] == question_owner]
        with st.spinner("Building a wardrobe-based recommendation..."):
            result, warning = recommend(
                selected_wardrobe,
                selected_question_event,
                question=question,
                style_first=False,
                use_fine_tuned=use_fine_tuned and not detected_garment,
                requested_article_type=detected_garment,
            )
        render_result(result, warning, selected_question_event, "Your ReWear response", wardrobe=selected_wardrobe)

# Entry point

def main() -> None:
    st.set_page_config(page_title="ReWear | Sustainable Stylist", layout="wide")
    st.markdown(load_app_css(), unsafe_allow_html=True)

    wardrobe = load_wardrobe()
    if wardrobe.empty:
        st.error("Scored wardrobe data could not be loaded. Run the data pipeline before launching ReWear.")
        st.stop()

    use_fine_tuned = render_sidebar(wardrobe)

    st.markdown("<div class='rw-shell'>", unsafe_allow_html=True)
    render_html("""
        <div class="rw-nav">
            <div class="rw-logo">ReWear</div>
            <div class="rw-nav-meta">Sustainability-first wardrobe stylist</div>
        </div>
    """)

    style_tab, ask_tab, wardrobe_tab = st.tabs(["Create a look", "Ask ReWear", "Explore wardrobe"])

    with style_tab:
        render_create_a_look_tab(wardrobe, use_fine_tuned)

    with ask_tab:
        render_ask_rewear_tab(wardrobe, use_fine_tuned)

    with wardrobe_tab:
        render_wardrobe_explorer(wardrobe)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()