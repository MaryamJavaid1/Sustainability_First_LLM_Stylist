# ReWear

A sustainability-first wardrobe recommendation system, built as an MSc
dissertation project. ReWear recommends outfits from clothes a user
already owns, favouring reuse and lower-impact materials over anything new
 and is explicit about the evidence and assumptions behind every
recommendation, rather than presenting proxy scores as measured facts.

## What's in this repo

- **`src/`**  the full data pipeline: cleaning the source catalogue,
  extracting material evidence, scoring garments, generating synthetic
  wardrobes, and building the fine-tuning dataset.
- **`app/`**  a Streamlit web app demonstrating both the rule-based
  recommendation engine and a locally-run fine-tuned model. See
  [`APP_README.md`](APP_README.md) for how to set it up and run it.
- **`notebooks/`**  the LoRA fine-tuning notebook, plus the saved
  held-out evaluation outputs it produced.
- **`tests/`**  the project's test suite.
- **`data/processed/wardrobe_scored.csv`**  the final scored dataset the
  whole system runs on.

## The short version of what it does

Starting from a general-purpose fashion product catalogue, the pipeline
extracts material evidence wherever it's genuinely available, scores
garments using a transparent, documented proxy rather than a black-box
number, and withholds scoring entirely for garments without defensible
evidence rather than guessing. That scored data feeds both a rule-based
recommendation engine and a LoRA-fine-tuned language model, which are
evaluated against each other on held-out prompts.

## Getting started

For running the web app specifically, see [`APP_README.md`](APP_README.md).

To run the data pipeline from scratch, the numbered scripts in `src/` are
meant to be run in order (`01_...` through `10_...`)  each one documents
its own inputs and outputs at the top of the file.
