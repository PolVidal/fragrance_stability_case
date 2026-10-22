# Fragrance Shelf-Life Stability

Root-cause analysis and predictive model of fragrance shelf-life stability from production sensor data
(259,200 batches, 5 plants, April–June 2023). Take-home case for the Symrise Data & AI Graduate Program.

The case statement is in [docs/usecase_statement.md](docs/usecase_statement.md).

## Results in short

**Root cause: two problems of different size**
- **Acute:** from the night of May 7–8, three reactors in Plant-04 (R03, R07, R08) lost about 3.3 months of
  shelf life overnight (−3.26 months, 95% CI −3.32 to −3.20, against the other Plant-04 reactors), together with
  a 4-SD drop in mixing torque. This event produces 18% of the Technical-grade batches.
- **Chronic:** the other 82% come from normal operation. In every plant and every week about 23% of batches end
  as Technical grade. Four operating conditions explain most of it: mixing torque, reaction temperature, pH
  (best around 5.45) and vessel pressure (best around 2.3 bar). The normal ranges in the statement are too wide.
- **No plant is worse than another** once the operating conditions are the same: plant gaps fall from 0.4–1.0
  months to about 0.01.
- Batches that already run at torque ≥ 12.2 N·m and ≤ 85 °C average 24.4 months instead of 21.7. This is an
  association, so the recommendation is a controlled trial before changing procedures.

**Model**
- LightGBM on six sensors, without density (measured only after separation), so the prediction is available
  earlier in the process.
- Test period (June 9–29, never used for any choice): mean absolute error 0.73 months, 0.50 on Technical batches.
  The reactor-average baseline gets 1.63.
- 90% prediction intervals by conformalized quantile regression, with about 90% coverage also for batches with
  missing sensors.
- Red / amber / green decision from the interval: the lab checks 42% of batches and still sees 99% of the
  Technical ones.
- Per-batch explanations in months (SHAP) and a warning when a sensor is outside the training range.

## Repository structure
```
data/raw/          fragrance_stability.parquet (converted from the provided CSV)
data/processed/    clean table written by notebook 02 (not committed)
docs/              case statement and model card
notebooks/         01_data_audit → 02_cleaning → 03_eda_root_cause → 04_model
reports/figures/   every figure, written by the notebooks
scripts/           CSV → Parquet conversion, benchmark of extra model families
src/fragrance/     reusable code: config, audit, cleaning, root cause, model, plots
tests/             unit tests (cleaning rules, root-cause helpers, model)
```

## How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                     # unit tests
jupyter lab                                # then run notebooks/01 → 04 in order
```
The notebooks are committed with their outputs, so they can be read without running them. Notebook 02 writes
`data/processed/batches_clean.parquet`, which notebooks 03 and 04 read. Notebook 04 retrains all candidate models
and takes about 5 minutes. Random seeds are fixed.

To rebuild the Parquet file from the original CSV:
`python3 scripts/convert_to_parquet.py path/to/fragrance_stability_dataset.csv`

## Where to look
- The four notebooks, in order: each section ends with its conclusions.
- [docs/model_card.md](docs/model_card.md): what the model does, how well, and its limits.

## Tools
I used Claude as support during development. It helped me with writing and structuring the code, while I designed
the pipeline, made the decisions and adjusted the analysis throughout.
The conclusions and recommendations are my own: I reviewed, validated and decided every one of them.
