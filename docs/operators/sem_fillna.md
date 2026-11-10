# `sem_fillna`: impute missing values in one column

`sem_fillna` fills the missing values of one column. It has two quite different modes:

* **Default (LLM + model):** the LLM proposes five imputation models. SemPipes cross-checks them on the rows where the value is known and trains the best one. Predictions come from that model, so imputing test data needs no LLM calls.
* **`with_llm_only=True` (per-row):** the LLM imputes every missing value directly from the other values in the same row, e.g. "infer the manufacturer from the product title". This uses world knowledge that a model trained on your data can't have, but it costs one request per missing value, every time.

| | Default mode | `with_llm_only=True` |
|---|---|---|
| Style | Code generation (model selection) | Per-row |
| LLM called | When fitting: 1 code-generation conversation. Never at transform time. | At transform time (training **and** test data): 1 request per missing value |
| LLM sees data values? | No: only column names, dtypes and your prompt | Yes: the whole row, plus candidate values |
| Optimisable | no | no |
| Source | `sempipes/operators/sem_fillna_llm_plus_model.py` | `sempipes/operators/sem_fillna_llm.py` |

The dispatch between the two modes happens in `sempipes/operators/__init__.py`.

## Signature

```python
DataOp.sem_fillna(
    target_column: str,
    nl_prompt: str,
    impute_with_existing_values_only: bool,
    **kwargs,          # with_llm_only: bool = False
) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `target_column` | `str` | required | Column whose missing values (`NaN`/`None`/`pd.NA`) are filled. |
| `nl_prompt` | `str` | required | Hints for the LLM, e.g. which columns are informative or what kind of method to use. |
| `impute_with_existing_values_only` | `bool` | required | **Default mode:** must be `True`, otherwise a bare `AssertionError` is raised; it has no other effect there. **Per-row mode:** `True` lists the observed values and asks the LLM to use only those; `False` shows them only as examples, so the LLM may invent new values (e.g. a manufacturer not yet in the column). Not enforced in either case. |
| `with_llm_only` | `bool` (keyword) | `False` | `True` switches to per-row mode. Any other keyword argument is silently ignored. |

## How it works

### Default mode: LLM proposes models, SemPipes picks the best one

At **fit** time:

1. The code-generation model is asked for **five very different** imputation models. They can be scikit-learn estimators, `IterativeImputer`, or AutoGluon's `TabularPredictor`. The prompt contains only the target column's name and dtype, whether it is numeric or categorical, the names of the other columns, your `nl_prompt` and the installed library versions. **No data values are sent.**
2. The answer may only *instantiate* models (`suggested_models_and_imputers = [model1, ..., model5]`). Code that calls `fit`, `predict` or `transform`, or uses `df`, is rejected and retried, as is code that fails to run.
3. All **other columns** are used as features. SemPipes turns every candidate into a predictor of the target column:
   * **scikit-learn models** (anything with `predict`) get a fixed feature preprocessing step. Numeric columns get median imputation plus standard scaling, and categorical columns get one-hot encoding. A pipeline that brings its own `ColumnTransformer` is used as is.
   * **Imputers** (e.g. `IterativeImputer`, `KNNImputer`) see the preprocessed features plus the target as the last column; categorical targets are encoded as category codes. To predict, the target column is set to missing and the imputer fills it in.
   * **AutoGluon `TabularPredictor`s** are trained by SemPipes on the raw DataFrame with a time limit of 60 seconds (the generated code may not call `fit` itself).
4. Each candidate is trained on 80% of the rows where the target is known and evaluated on the remaining 20% (`random_state=42`), using RMSE for numeric targets and error rate for categorical ones.
5. The best candidate is refitted on all known rows. AutoGluon predictors can't be refitted, so the one trained for evaluation is kept, and the same happens if refitting another candidate fails. If no candidate works (or code generation failed completely), SemPipes falls back to `DummyRegressor(strategy="median")` or `DummyClassifier(strategy="most_frequent")` and prints a message.

At **transform** time, the fitted model predicts the missing values. Other rows are not touched.

**About AutoGluon:** each AutoGluon candidate costs up to a minute of training. Asking for "a very fast method" or "only scikit-learn models" in `nl_prompt` steers the LLM towards cheap candidates.

### `with_llm_only=True`: one LLM request per missing value

At **fit** time, nothing is sent to the LLM. SemPipes only records the dtype and the observed values of the target column.

At **transform** time (so also inside `fit_transform`, on training data), SemPipes builds one request per row with a missing value and sends it to the batch model (`llm_for_batch_processing`) in chunks of `batch_size_for_batch_processing`. Each request contains:

* the target column name and dtype;
* **all other values of that row** (as a `{column: value}` dict);
* up to 99 candidate values. If the column has more than 99 distinct values, SemPipes narrows the list per row using values that co-occurred with the row's other values in the training data, or else uses the 99 most frequent;
* your `nl_prompt` and the existing-values instruction.

The answer is stripped of whitespace and written into the column **as a string**. A failed request is not retried: one failed request makes the whole step raise.

## Examples

Per-row mode on a tiny table (3 LLM requests):

```python
import pandas as pd
import skrub

import sempipes

products = pd.DataFrame(
    {
        "title": ["iPhone 15 Pro 256GB", "Galaxy S24 Ultra", "MacBook Air M3 13-inch",
                  "Pixel 8 128GB", "ThinkPad X1 Carbon Gen 11", "iPad mini (6th generation)"],
        "make": ["Apple", "Samsung", None, "Google", None, None],
    }
)
data = skrub.var("products", products)

filled = data.sem_fillna(
    target_column="make",
    nl_prompt="Infer the manufacturer from the product title.",
    impute_with_existing_values_only=False,   # allow new values such as "Lenovo"
    with_llm_only=True,
)
print(filled.skb.preview())
```

Default mode (an LLM-chosen model learns from the rows where the value is known):

```python
import skrub

import sempipes

salaries = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries.head(500).copy()
salaries.loc[salaries.sample(frac=0.2, random_state=0).index, "department"] = None

data = skrub.var("salaries", salaries)
imputed = data.sem_fillna(
    target_column="department",
    nl_prompt="Infer the department from the job title and division. Use a very fast method.",
    impute_with_existing_values_only=True,   # required in default mode
)
print(imputed.skb.preview()["department"].isna().sum())   # 0
```


## See also

* Example script: [`examples/04_sem_fillna.py`](../../examples/04_sem_fillna.py)
* Notebook: [`examples/notebooks/demo__sem_fillna.ipynb`](../../examples/notebooks/demo__sem_fillna.ipynb)
* Related: [`sem_clean`](sem_clean.md) (general cleaning, including simple imputation) and [`sem_extract_features`](sem_extract_features.md) (deriving *new* columns from text and images).
