# Sustainability-First Stylist Web App

A Streamlit app that recommends outfits from a wardrobe you already own,
prioritising reuse and lower-impact materials. It runs on top of the data
pipeline's output (`data/processed/wardrobe_scored.csv`).

## What it does

Three tabs:

1. **Create a look** pick a wardrobe and an occasion, then choose how
   outfits get picked: sustainability-first (ranks by the environmental and
   circularity proxy score), least-worn comparison (ranks by lowest prior
   wear count), or compare both side by side. You get a full outfit, a
   rationale, an evidence panel, and a CO2e production-impact estimate.
   There's also a toggle to switch the sustainability-first pick over to a
   fine-tuned local language model instead of the rule-based engine.
2. **Ask ReWear**  type a request in plain language (e.g. "I have a formal
   dinner, I want to wear a saree"). The app figures out the occasion and,
   if you mentioned one, a specific garment type too. If you asked for a
   specific type, it gives you the single best match of that type rather
   than a full outfit  and if the wardrobe genuinely doesn't have one, it
   says so plainly instead of quietly picking something else.
3. **Explore wardrobe**  filter and sort every scored garment in the
   dataset by season, category, material, wearability, and minimum score.

Whenever a constraint (season, formality, or a requested garment type)
can't be fully met, the app tells you rather than silently substituting
something that doesn't actually match.

## Setup

From the project root:

```bash
# 1. Install app dependencies
pip install -r requirements-app.txt

# 2. Make sure the pipeline's been run at least once, so the scored
#    wardrobe exists
ls data/processed/wardrobe_scored.csv     # must exist

# 3. Make sure garment images are there (see below if not)
ls app/assets/garments/ # should contain .jpg files

# 4. Run it
streamlit run app/streamlit_app.py
```

It opens automatically in your browser (usually http://localhost:8501).

## Regenerating garment images

`app/assets/garments/` (~5.9 GB, ~15,500 images) isn't tracked in git  it's
fully regenerable, so there's no point storing it. To rebuild it on a fresh
clone:

1. Download the Kaggle Fashion Product Images dataset and put it at
   `data/raw/fashion_product_images_full/images/` (also gitignored).
2. Run the pipeline through `src/06_calculate_final_sustainability.py` to
   produce `data/processed/wardrobe_scored.csv`.
3. Run `python -m src.10_collect_wardrobe_images` to copy just the images
   the scored wardrobe actually references (a small slice of the full
   ~44k-image Kaggle set) into `app/assets/garments/`.

## Enabling the fine-tuned model (optional)

The app works fully without this  it defaults to the rule-based engine,
which mirrors the pipeline's own outfit-selection logic. If you want to try
the actual fine-tuned model locally:

1. Install [Ollama](https://ollama.com) and make sure it's running
   (`ollama serve`).
2. The `Modelfile` in `rewear-ollama/` is tracked in git and already has the
   right system prompt and generation settings  you just need the actual
   model weights, which aren't tracked because the file is several GB.
   Export these from the fine-tuning notebook (see the GGUF export cell),
   drop the resulting `.gguf` file into `rewear-ollama/`, then build it:
   ```bash
   cd rewear-ollama
   ollama create rewear-sustainability -f Modelfile
   ```
3. Restart the app. A **"Use fine-tuned model"** toggle will appear in the
   sidebar.

When it's on, the fine-tuned model picks the garments itself for
sustainability-first requests  not just the explanation text. It only
applies to full-outfit requests; if you ask for a specific garment type,
the app always falls back to the rule-based engine, since the fine-tuned
model was never evaluated on that kind of request.

## How this relates to the dissertation's evaluation

This local setup runs the same fine-tuned model that's evaluated in the
dissertation, but through a different, quantised local pipeline (via
llama.cpp/Ollama) than the one used to produce the actual reported
evaluation numbers. The two don't always behave identically, and that
difference is investigated properly in the write-up rather than ignored.
If you want the real evaluation results, those come from the fine-tuning
notebook and its saved evaluation outputs  not from playing with this app.

## Files

- `app/streamlit_app.py`  the web interface: three tabs, sidebar, all page
  rendering.
- `app/style.css`  the interface's stylesheet, loaded at runtime.
- `app/recommender.py`  the actual recommendation logic: the rule-based
  engine (shared with `08_generate_prompts.py`), garment-type request
  parsing, and the local fine-tuned model integration via Ollama, all
  behind one `recommend()` entry point that handles fallback and
  constraint-relaxation disclosure.