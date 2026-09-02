# Makefile single entrypoint for the sustainability-stylist pipeline.
#
# Usage:
#   make all        Run the full pipeline in order
#   make inspect     01
#   make clean-data  02
#   make materials   03b
#   make score       04 + 04b + 04c
#   make wardrobe    05 + 06
#   make outputs     07 + 08 + 09
#   make clean       Remove __pycache__ and *.pyc (does NOT touch data/)

PYTHON := python -m

.PHONY: all inspect clean-data materials score wardrobe outputs clean

all: inspect clean-data materials score wardrobe outputs

inspect:
	$(PYTHON) src.01_inspect_primary_dataset

clean-data:
	$(PYTHON) src.02_clean_apparel_metadata

materials:
	$(PYTHON) src.03b_extract_full_json_materials

score:
	$(PYTHON) src.04_apply_lca_scores
	$(PYTHON) src.04b_create_material_audit
	$(PYTHON) src.04c_create_manual_review_sample

wardrobe:
	$(PYTHON) src.05_generate_wardrobe_attributes
	$(PYTHON) src.06_calculate_final_sustainability

outputs:
	$(PYTHON) src.07_visualize_data
	$(PYTHON) src.08_generate_prompts
	$(PYTHON) src.09_generate_baseline_prompts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete