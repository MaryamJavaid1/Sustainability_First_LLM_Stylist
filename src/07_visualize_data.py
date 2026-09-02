"""07_visualize_data.py

Generates distribution charts describing the sustainability scoring output
for the dissertation's methodology/results chapters.

Note: these charts describe the distribution and behaviour of the scores
produced by the formula in 06_calculate_final_sustainability.py. They do not
mathematically validate the formula itself validation would require
sensitivity analysis, comparison against an external benchmark, expert
review, or testing against real wardrobe data (see Limitations).

Input:
    data/processed/wardrobe_scored.csv

Outputs:
    outputs/sustainability_score_distribution.png
    outputs/material_scores_by_category.png
    outputs/reuse_vs_condition.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import OUTPUTS, PROCESSED_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"
TOP_N_CATEGORIES = 5
CONDITION_ORDER = ["Excellent", "Good", "Fair", "Worn"]


def generate_visualizations() -> None:
    """Generate distribution charts describing the scored wardrobe dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing scored dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    sns.set_theme(style="whitegrid")

    # 1. Distribution of final sustainability scores.
    plt.figure(figsize=(10, 6))
    sns.histplot(df["sustainability_score"], bins=20, kde=True, color="teal")
    plt.title("Distribution of Sustainability Scores Across Wardrobe Pilot")
    plt.xlabel("Sustainability Score (out of 10.0)")
    plt.ylabel("Number of Garments")
    plt.savefig(OUTPUTS / "sustainability_score_distribution.png")
    plt.close()

    # 2. Material scores by top article types.
    top_categories = df["article_type"].value_counts().head(TOP_N_CATEGORIES).index
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df[df["article_type"].isin(top_categories)],
        x="article_type",
        y="material_score",
        hue="article_type",
        palette="Set2",
        legend=False,
    )
    plt.title("Material Impact Scores by Top Article Types")
    plt.xlabel("Article Type")
    plt.ylabel("Material Score (Higher = More Sustainable)")
    plt.savefig(OUTPUTS / "material_scores_by_category.png")
    plt.close()

    # 3. Reuse count vs. condition.
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="condition",
        y="reuse_count",
        hue="condition",
        order=CONDITION_ORDER,
        palette="YlOrRd",
        legend=False,
    )
    plt.title("Relationship Between Synthetic Reuse Count and Condition")
    plt.xlabel("Garment Condition")
    plt.ylabel("Number of Times Worn (Reuse Count)")
    plt.savefig(OUTPUTS / "reuse_vs_condition.png")
    plt.close()

    logger.info("Visualisations saved to: %s", OUTPUTS)


if __name__ == "__main__":
    generate_visualizations()