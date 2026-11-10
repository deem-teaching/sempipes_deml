# `sem_gen_features`: generate new feature columns with LLM-written code

`sem_gen_features` asks the LLM to write feature-engineering code for a table. The new columns combine or transform existing ones using domain knowledge, e.g. "age of the house at sale", "total number of bathrooms" or "price per square metre". It follows the CAAFE approach (Context-Aware Automated Feature Engineering) and builds on Apache-licensed code from [CAAFE](https://github.com/noahho/CAAFE) (see `NOTICE`). The original columns are kept and the new ones are appended.

| | |
|---|---|
| Style | Code generation |
| LLM called | When fitting: 1 code-generation conversation. Never at transform time. |
| Optimisable | **yes** (see [optimizer](../optimizer.md)) |
| Source | `sempipes/operators/sem_gen_features_caafe.py` |

## Signature

```python
DataOp.sem_gen_features(
    nl_prompt: str,
    name: str,
    how_many: int = 10,
    n_prompt_samples: int = 10,
) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `nl_prompt` | `str` | required | What kind of features to create and any domain knowledge to use. |
| `name` | `str` | required | Unique name of this operator in the pipeline (used by the optimiser and for the hidden variables). Must not contain `__`. |
| `how_many` | `int` | `10` | Upper limit on new columns that is stated in the prompt ("up to `how_many` new feature columns"). It is not enforced. |
| `n_prompt_samples` | `int` | `10` | How many rows (the **first** `n_prompt_samples` rows, not a random sample) are shown to the LLM per column. Must be ≥ 1, otherwise `ValueError`. |

## How it works

At **fit** time, the code-generation model receives:

* for every column: name, dtype, percentage of missing values and the first `n_prompt_samples` values;
* the number of training rows;
* a task description. **Outside the optimiser** this is generic: "a downstream classification algorithm (such as XGBoost)" evaluated with "accuracy". **Under [`optimise_colopro`](../optimizer.md)** it uses the pipeline summary instead: the model class, classification or regression, the target's name and `as_y` description, the dataset's `as_X` description and the scoring metric;
* your `nl_prompt`;
* during optimisation, also earlier code versions with their scores (memory) and up to three high-scoring "inspiration" snippets.

The LLM must answer with `_sem_gen_features(df) -> pandas.DataFrame` and explain each feature in comments. SemPipes runs it on the **first 100 rows** and checks that it returns a DataFrame with the same number of rows and **all original columns**. Errors are fed back, up to `max_retries_for_code_gen` attempts.

At **transform** time, the stored function runs on a copy of the input. If it raises, SemPipes writes the code and the stack trace to `.sem_gen_features_error_<timestamp>/` in the working directory and re-raises the error.

**If every attempt fails**, SemPipes logs an error and falls back to an identity function, so **no features are added**.

## Example

```python
import pandas as pd
import skrub

import sempipes

houses = pd.DataFrame(
    {
        "year_built": [1995, 2005, 1970, 2015, 1988, 1960, 2001, 1999],
        "year_sold": [2010, 2010, 2008, 2020, 2009, 2007, 2015, 2012],
        "full_bath": [2, 3, 1, 2, 1, 1, 2, 2],
        "half_bath": [1, 0, 1, 1, 0, 0, 1, 0],
        "living_area_sqft": [1800, 2400, 1100, 2000, 950, 1200, 1700, 1600],
        "lot_area_sqft": [8000, 9500, 6000, 7200, 5000, 7000, 8800, 6400],
    }
)
data = skrub.var("houses", houses)

with_features = data.sem_gen_features(
    nl_prompt="Compute features that help to predict the sale price, e.g. the age of the house at sale "
    "and the total number of bathrooms.",
    name="house_features",
    how_many=3,
)
print(with_features.skb.preview())
```

In a full pipeline, put `sem_gen_features` **after** `mark_as_X()` so that it is fitted per cross-validation fold (see [concepts](../concepts.md#traintest-discipline)):

```python
from sklearn.ensemble import HistGradientBoostingRegressor

X = sempipes.as_X(skrub.var("houses"), "Residential houses sold in Ames, Iowa")
y = sempipes.as_y(skrub.var("price"), "Sale price in USD")
X = X.sem_gen_features(nl_prompt="...", name="house_features", how_many=5)
pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(HistGradientBoostingRegressor(), y=y)
```

## See also

* Example script: [`examples/07_sem_gen_features.py`](../../examples/07_sem_gen_features.py)
* Notebooks: [`examples/notebooks/demo__optimise_semantic_operator.ipynb`](../../examples/notebooks/demo__optimise_semantic_operator.ipynb), [`examples/notebooks/demo__optimise_multiple_semantic_operators.ipynb`](../../examples/notebooks/demo__optimise_multiple_semantic_operators.ipynb)
* Related: [`sem_agg_features`](sem_agg_features.md) (features from a *second* table), [`sem_extract_features`](sem_extract_features.md) (features from text and images), [optimizer](../optimizer.md).
