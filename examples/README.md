# SemPipes examples

One short, standalone script per semantic operator. Each script is a small tutorial: read it from top to bottom
(the docstring explains what the operator does and what the example shows), run it, and compare the printed
output with the code.

## How to run

From the repository root:

```bash
poetry install                       # once
cp .env.example .env                 # once, then put your API key(s) into .env
poetry run python examples/01_sem_select.py
```

- `import sempipes` loads `.env` automatically. By default every operator uses `gemini/gemini-2.5-flash` through
  [litellm](https://docs.litellm.ai/), so you need a Gemini API key. To use another provider, call
  `sempipes.update_config(llm_for_code_generation=sempipes.LLM("openai/gpt-4.1"), llm_for_batch_processing=...)`
  at the top of a script. To use SAIA (GWDG Academic Cloud), add `sempipes._set_config(sempipes.get_default_saia_config())`
  after `import sempipes` (see [configuration.md](../docs/configuration.md#using-saia-gwdg-academic-cloud)).
- The first `import sempipes` takes a while, because it imports torch and AutoGluon. The skrub datasets are
  downloaded to `~/skrub_data` on first use.
- The scripts resolve their data paths relative to their own location, so they also work from other directories.
- LLM answers vary between runs. Generated features or synthetic rows do not always improve the model, and seeing
  when they do not is part of the lesson.

## Overview

Runtimes are rough wall-clock estimates on a laptop and depend mostly on LLM latency. "LLM calls" counts requests to
the model. Code-generating operators may add up to 4 retries when their generated code fails validation.

| Script | Operator | Dataset | Runtime | LLM calls |
|---|---|---|---|---|
| [`01_sem_select.py`](01_sem_select.py) | [`sem_select`](../docs/operators/sem_select.md) | skrub `employee_salaries` (9228 rows) | ~1 min | 8 (one per column) |
| [`02_sem_clean.py`](02_sem_clean.py) | [`sem_clean`](../docs/operators/sem_clean.md) | inline table (6 rows) | < 1 min | 1 |
| [`03_sem_refine.py`](03_sem_refine.py) | [`sem_refine`](../docs/operators/sem_refine.md) | inline addresses (16 rows) | < 1 min | 2 (one per column) |
| [`04_sem_fillna.py`](04_sem_fillna.py) | [`sem_fillna`](../docs/operators/sem_fillna.md), both modes | `data/restaurants` + 600 rows of `employee_salaries` | 2-3 min | 30 + 1 |
| [`05_sem_extract_features_text.py`](05_sem_extract_features_text.py) | [`sem_extract_features`](../docs/operators/sem_extract_features.md) (text) | skrub `toxicity` (100 tweets) | 1-2 min | 100 (batches of 20) |
| [`06_sem_extract_features_images.py`](06_sem_extract_features_images.py) | [`sem_extract_features`](../docs/operators/sem_extract_features.md) (images) | `data/fashion` (12 product photos) | < 1 min | 12 (multimodal) |
| [`07_sem_gen_features.py`](07_sem_gen_features.py) | [`sem_gen_features`](../docs/operators/sem_gen_features.md) | `data/house_prices` (462 houses) | 1-2 min | 1 |
| [`08_sem_agg_features.py`](08_sem_agg_features.py) | [`sem_agg_features`](../docs/operators/sem_agg_features.md) | skrub `credit_fraud` (3000 baskets) | 1-2 min | 1 |
| [`09_sem_augment.py`](09_sem_augment.py) | [`sem_augment`](../docs/operators/sem_augment.md) | `employee_salaries` (1000 rows + 500 synthetic) | 1-3 min | 1 |
| [`10_sem_distill.py`](10_sem_distill.py) | [`sem_distill`](../docs/operators/sem_distill.md) | `employee_salaries` (8000 → 800 rows) | 1-2 min | 1 |
| [`11_sem_choose.py`](11_sem_choose.py) | [`sem_choose`](../docs/operators/sem_choose.md) | skrub `midwest_survey` (2494 rows) | 2-3 min | 2 (one per parameter) |
| [`12_optimise_colopro.py`](12_optimise_colopro.py) | [`optimise_colopro`](../docs/optimizer.md) | `midwest_survey` (1200 rows) | 4-10 min | ~8-15 |

Suggested order: 01-04 (cleaning and preparing data), 05-08 (creating features), 09-10 (changing the training
set), 11-12 (tuning and optimisation).

## Patterns you will see in every script

- **`import sempipes`** adds the `.sem_*` methods to skrub `DataOp`s. The operators are ordinary steps of a skrub
  pipeline, so the usual skrub tools work on them: `make_learner()`, `train_test_split()`, `make_randomized_search()`
  and so on.
- **Fit once, reuse.** Most operators call the LLM while the pipeline is *fitted* and store the result (generated
  code, selected columns, a fitted model). Predicting on new data reuses the stored result without new LLM calls.
  The exceptions are per-row operators (`sem_fillna(..., with_llm_only=True)` and
  `sem_extract_features(..., generate_via_code=False)`), which call the LLM for every row they process.
  `sem_choose` calls the LLM right away, while you define the pipeline.
- **Variables without values.** The scripts create inputs with `skrub.var("name")` and pass the data later as a
  dictionary, e.g. `learner.fit({"houses": X_train, ...})`. This keeps skrub from running (and paying for) LLM-backed
  previews while the pipeline is being built.
- **Hidden helper variables.** Operators such as `sem_gen_features` also register variables like
  `sempipes_prefitted_state__<name>`, which the optimiser uses. `pipeline.skb.get_data()` returns their default
  values, so the scripts start from that dictionary and add their own data. Script 12 shows how to set the prefitted
  state to reuse optimised code.

## Cost

- Running all twelve scripts once makes roughly 200 LLM requests. Most of them are short per-row requests from
  scripts 04, 05 and 06. With the default Gemini 2.5 Flash model this typically costs well below one US dollar, but
  check your provider's current pricing.
- Per-row operators scale linearly with the number of rows: 10 000 rows means 10 000 requests. That is why these
  examples use at most 100 rows there. Code-generating operators cost about the same regardless of data size.
- SemPipes logs `TOKEN_USAGE,<input tokens>,<output tokens>` for every request, so you can add up the usage of a run.
- With a free-tier API key you may hit rate limits in the per-row scripts. Lower the number of parallel requests
  with `sempipes.update_config(batch_size_for_batch_processing=5)`.

## Notebooks

The notebooks in [`notebooks/`](notebooks/) are shorter, interactive demos. Start Jupyter with
`poetry run jupyter notebook examples/notebooks/`.

| Notebook | What it shows |
|---|---|
| [`demo.ipynb`](notebooks/demo.ipynb) | An end-to-end fraud-detection pipeline on `credit_fraud` that combines `sem_fillna`, `sem_gen_features` and `sem_choose`. |
| [`demo__sem_select.ipynb`](notebooks/demo__sem_select.ipynb) | `sem_select` on a tiny table: selecting columns by meaning (paper sizes, metric units). |
| [`demo__sem_fillna.ipynb`](notebooks/demo__sem_fillna.ipynb) | LLM-only imputation of product manufacturers in `credit_fraud`. |
| [`demo__sem_agg_features.ipynb`](notebooks/demo__sem_agg_features.ipynb) | `sem_agg_features`: aggregating the products of each shopping basket into basket features. |
| [`demo_sem_augment.ipynb`](notebooks/demo_sem_augment.ipynb) | `sem_augment`: generating additional, targeted product rows. |
| [`demo_sem_extract_features.ipynb`](notebooks/demo_sem_extract_features.ipynb) | `sem_extract_features` on tweets with explicit output columns, in code-generation mode. |
| [`demo__optimise_semantic_operator.ipynb`](notebooks/demo__optimise_semantic_operator.ipynb) | `optimise_colopro` improving a single `sem_gen_features` operator on the Midwest survey. |
| [`demo__optimise_multiple_semantic_operators.ipynb`](notebooks/demo__optimise_multiple_semantic_operators.ipynb) | Optimising several semantic operators of one pipeline together. |
| [`demo_view_trajectory.ipynb`](notebooks/demo_view_trajectory.ipynb) | Loading and visualising a saved optimisation trajectory from `.sempipes_trajectories/`. |
