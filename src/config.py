"""config.py

Central configuration for the sustainability-stylist pipeline.

All other pipeline scripts import paths and constants from here instead of
hard-coding them. This is the single place to change when moving the project
to another machine, another dataset location, or another random seed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Absolute root of the project (two levels up from this file: src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from the .env file in the project root, if present.
load_dotenv(PROJECT_ROOT / ".env")


# Data directories

RAW_DATA = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
INTERIM_DATA = PROJECT_ROOT / os.getenv("INTERIM_DATA_DIR", "data/interim")
PROCESSED_DATA = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
OUTPUTS = PROJECT_ROOT / os.getenv("OUTPUTS_DIR", "outputs")


# Dataset folder names

PRIMARY_FOLDER = os.getenv("PRIMARY_DATASET_FOLDER", "fashion_product_images_full")

# NOTE: the secondary sustainability dataset (Waqar Ali, Kaggle) was withdrawn
# after usage permission was declined by the dataset owner. It is intentionally
# not referenced anywhere in this pipeline. See Chapter 3, "Deviations from
# the Approved Proposal", in the dissertation for the full rationale.


# Reproducibility

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


# Ensure output folders exist before any script writes to them.

for _directory in (INTERIM_DATA, PROCESSED_DATA, OUTPUTS):
    _directory.mkdir(parents=True, exist_ok=True)