"""06c_sensitivity_analysis.py

Weight sensitivity analysis for the composite sustainability score.

The composite score combines four component scores with fixed weights
(material 0.4444, reuse 0.2778, wearability 0.1667, second_hand 0.1111).
These weights are a transparent project design choice, not values published
by any single source. This stage tests how sensitive the RESULTS are to that
choice, by recomputing the score under alternative weightings and measuring
whether the highest-ranked garments change.

Scenarios
-
- baseline        : the current weights.
- material_focus  : material weighted more heavily.
- reuse_focus     : reuse weighted more heavily.
- equal_weights   : all four components weighted equally.

For each scenario the score is recomputed and the top-N ranked garments are
compared against the baseline using rank-overlap (how many of the baseline
top-N remain in the scenario top-N) and Spearman rank correlation over all
garments. High overlap / high correlation => the method is robust to the
weighting choice; large changes => the result is sensitive and this is
reported as a limitation.

Input:  data/processed/wardrobe_scored.csv
Output: outputs/sensitivity_analysis.csv        (summary table)
        outputs/sensitivity_top_overlap.csv     (top-N overlap detail)
"""

from __future__ import annotations

import pandas as pd

from src.config import OUTPUTS, PROCESSED_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"
SUMMARY_FILE = OUTPUTS / "sensitivity_analysis.csv"
OVERLAP_FILE = OUTPUTS / "sensitivity_top_overlap.csv"

COMPONENTS = ["material_score", "reuse_score", "wearability_score", "second_hand_bonus"]

# Each scenario's raw weights (need not sum to 1; we renormalise below).
SCENARIOS: dict[str, dict[str, float]] = {
    "baseline":       {"material_score": 0.4444, "reuse_score": 0.2778,
                       "wearability_score": 0.1667, "second_hand_bonus": 0.1111},
    "material_focus": {"material_score": 0.60, "reuse_score": 0.20,
                       "wearability_score": 0.13, "second_hand_bonus": 0.07},
    "reuse_focus":    {"material_score": 0.25, "reuse_score": 0.50,
                       "wearability_score": 0.15, "second_hand_bonus": 0.10},
    "equal_weights":  {"material_score": 0.25, "reuse_score": 0.25,
                       "wearability_score": 0.25, "second_hand_bonus": 0.25},
}

TOP_N = 100  # size of the "top recommendations" set to compare


def _weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total_w = sum(weights.values())
    score = sum(df[c] * (weights[c] / total_w) for c in COMPONENTS)
    return score.clip(lower=1.0, upper=10.0)


def run_sensitivity() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 06_calculate_final_sustainability.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    missing = [c for c in COMPONENTS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing component columns: {missing}")

    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=COMPONENTS).reset_index(drop=True)

    # Compute each scenario's score.
    scores = {name: _weighted_score(df, w) for name, w in SCENARIOS.items()}
    baseline = scores["baseline"]
    baseline_top = set(baseline.sort_values(ascending=False).head(TOP_N).index)

    summary_rows = []
    overlap_rows = []
    for name, s in scores.items():
        # Spearman rank correlation with baseline (over ALL garments).
        rho = baseline.corr(s, method="spearman")
        # Top-N overlap.
        top = set(s.sort_values(ascending=False).head(TOP_N).index)
        overlap = len(baseline_top & top)
        overlap_pct = round(100 * overlap / TOP_N, 1)

        summary_rows.append({
            "scenario": name,
            "mean_score": round(float(s.mean()), 3),
            "std_score": round(float(s.std()), 3),
            "spearman_vs_baseline": round(float(rho), 4),
            f"top{TOP_N}_overlap_pct_vs_baseline": overlap_pct,
        })
        overlap_rows.append({
            "scenario": name,
            f"top{TOP_N}_kept_from_baseline": overlap,
            f"top{TOP_N}_total": TOP_N,
            f"top{TOP_N}_overlap_pct": overlap_pct,
        })

    summary = pd.DataFrame(summary_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_FILE, index=False)
    overlap_df.to_csv(OVERLAP_FILE, index=False)

    logger.info("Weight sensitivity analysis completed.")
    logger.info("Garments analysed: %s", f"{len(df):,}")
    logger.info("Top-N compared: %s", TOP_N)
    logger.info("Summary:\n%s", summary.to_string(index=False))
    logger.info("Summary saved to: %s", SUMMARY_FILE)


if __name__ == "__main__":
    run_sensitivity()