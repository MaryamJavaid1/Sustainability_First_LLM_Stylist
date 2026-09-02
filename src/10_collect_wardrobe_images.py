"""10_collect_wardrobe_images.py

Copies ONLY the product images referenced by the final scored wardrobe
(data/processed/wardrobe_scored.csv) into a lean assets folder used by the
web interface.

The full Kaggle dataset ships ~44k images; the scored wardrobe uses far fewer.
Copying just the needed subset keeps the app fast and the repo small, while
leaving the read-only raw dataset untouched.

Input:
    data/processed/wardrobe_scored.csv
    data/raw/fashion_product_images_full/images/<garment_id>.jpg

Output:
    app/assets/garments/<garment_id>.jpg
"""

from __future__ import annotations

import shutil

import pandas as pd

from src.config import OUTPUTS, PRIMARY_FOLDER, PROCESSED_DATA, RAW_DATA
from src.logging_utils import get_logger

logger = get_logger(__name__, log_dir=OUTPUTS)

INPUT_FILE = PROCESSED_DATA / "wardrobe_scored.csv"
SOURCE_IMAGE_DIR = RAW_DATA / PRIMARY_FOLDER / "images"

# app/assets/garments lives next to the web app, not in data/, because these
# are presentation assets for the interface rather than pipeline data.
PROJECT_ROOT = RAW_DATA.parents[1]  # data/raw -> project root
DEST_IMAGE_DIR = PROJECT_ROOT / "app" / "assets" / "garments"


def collect_wardrobe_images() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Scored wardrobe not found: {INPUT_FILE}\n"
            "Run 06_calculate_final_sustainability.py first."
        )
    if not SOURCE_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Source image directory not found: {SOURCE_IMAGE_DIR}")

    DEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    garment_ids = (
        df["garment_id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).unique()
    )

    copied = 0
    missing = 0

    for garment_id in garment_ids:
        source = SOURCE_IMAGE_DIR / f"{garment_id}.jpg"
        dest = DEST_IMAGE_DIR / f"{garment_id}.jpg"

        if dest.exists():
            copied += 1
            continue

        if source.exists():
            shutil.copy2(source, dest)
            copied += 1
        else:
            missing += 1

    logger.info("Wardrobe image collection completed.")
    logger.info("Unique garments in scored wardrobe: %s", f"{len(garment_ids):,}")
    logger.info("Images available/copied: %s", f"{copied:,}")
    logger.info("Images missing from source: %s", f"{missing:,}")
    logger.info("Images saved to: %s", DEST_IMAGE_DIR)


if __name__ == "__main__":
    collect_wardrobe_images()