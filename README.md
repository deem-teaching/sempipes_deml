# SemPipes - Optimizable Semantic Data Operators for Tabular Machine Learning Pipelines

SemPipes adds LLM-powered **semantic operators** (`sem_*`) to [skrub](https://github.com/skrub-data/skrub) DataOps pipelines. You describe a data-preparation step in natural language, such as "infer the manufacturer from the product title" or "compute the age of the house at sale". The LLM then either writes Python code for that step once, from your training data, or processes the rows directly. The operators behave like scikit-learn transformers inside a skrub pipeline: they are fitted on training data, re-applied to new data, and several of them can be improved automatically by an LLM-driven optimiser.

SemPipes is a research project that relies heavily on the awesome work of the [skrub](https://github.com/skrub-data/skrub) project!

## Quickstart

```python
import pandas as pd
import skrub

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

products = pd.DataFrame({
    "title": ["iPhone 15 Pro 256GB", "Galaxy S24 Ultra", "MacBook Air M3 13-inch",
              "Pixel 8 128GB", "ThinkPad X1 Carbon Gen 11", "iPad mini (6th generation)"],
    "make": ["Apple", "Samsung", None, "Google", None, None],
})

data = skrub.var("products", products)           # a skrub variable with example data
filled = data.sem_fillna(
    target_column="make",
    nl_prompt="Infer the manufacturer from the product title.",
    impute_with_existing_values_only=False,       # allow new values such as "Lenovo"
    with_llm_only=True,                           # ask the LLM once per missing value
)
print(filled.skb.preview())                      # .skb.eval() would re-run the step (and the LLM calls)
```

skrub computes previews eagerly, so the three LLM requests already happen when `filled` is created. [Core concepts](docs/concepts.md) explains how to turn this into a pipeline that you fit on training data and apply to test data.

## Installation (short version)

You need **Python 3.11 or 3.12** (3.13 is not supported), **Poetry 2**, and an LLM API key. The default model is Gemini 2.5 Flash, and Google AI Studio offers free keys.

```bash
poetry env use python3.12
poetry install                                   # several GB: torch, transformers, autogluon, sdv
cp .env.example .env                             # then set GEMINI_API_KEY (or OPENAI_API_KEY)
poetry run python -c "import sempipes"           # the first import is slow
```

Details, other providers (OpenAI, SAIA of GWDG Academic Cloud, local models via Ollama) and troubleshooting: [docs/installation.md](docs/installation.md).

## Operators

| Operator | What it does | Optimisable |
|---|---|---|
| [`sem_select`](docs/operators/sem_select.md) | Keep the columns that match a description | no |
| [`sem_clean`](docs/operators/sem_clean.md) | Clean selected columns with LLM-written code | no |
| [`sem_refine`](docs/operators/sem_refine.md) | Normalise and deduplicate the values of one column (typos, synonyms, languages) | no |
| [`sem_fillna`](docs/operators/sem_fillna.md) | Impute missing values with an LLM-chosen model or per-row LLM calls | no |
| [`sem_extract_features`](docs/operators/sem_extract_features.md) | Extract features from text, image and audio columns | yes (code mode) |
| [`sem_gen_features`](docs/operators/sem_gen_features.md) | Generate new feature columns from existing ones (CAAFE-style) | yes |
| [`sem_agg_features`](docs/operators/sem_agg_features.md) | Aggregate a second table and join the result as features | yes |
| [`sem_augment`](docs/operators/sem_augment.md) | Add synthetic training rows | yes (code mode) |
| [`sem_distill`](docs/operators/sem_distill.md) | Reduce the training data to fewer, representative rows | no |
| [`sem_choose`](docs/operators/sem_choose.md) | Let the LLM propose hyperparameter search spaces | no |

Optimisable operators can be improved with [`optimise_colopro`](docs/optimizer.md).

## Learn more

* [Documentation index](docs/README.md): installation, core concepts, configuration, one page per operator, optimiser.
* [Examples](examples/README.md): one runnable script per operator, e.g. `poetry run python examples/01_sem_select.py`, plus notebooks in [`examples/notebooks/`](examples/notebooks/).

## License and acknowledgements

SemPipes is licensed under the [Apache License 2.0](LICENSE). It builds on [skrub](https://github.com/skrub-data/skrub), and `sem_gen_features` builds on Apache-licensed code from [CAAFE](https://github.com/noahho/CAAFE) by Prior Labs (see [NOTICE](NOTICE)).
