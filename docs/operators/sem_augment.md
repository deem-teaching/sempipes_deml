# `sem_augment`: add synthetic training rows

`sem_augment` appends `number_of_rows_to_generate` synthetic rows to your **training data**, e.g. to get more examples of a rare class. It has two modes:

* **Code mode (default, `generate_via_code=True`):** the LLM writes code that fits an [SDV](https://docs.sdv.dev/) synthesizer (e.g. `GaussianCopulaSynthesizer`, optionally conditioned on column values) and samples new rows. This mode is **optimisable**.
* **Direct mode (`generate_via_code=False`):** the LLM writes the new rows itself as JSON, up to 100 rows per request.

Augmentation only happens when fitting (`fit_transform`). At transform time (test data, predictions) the data passes through **unchanged**.

| | Code mode (default) | Direct mode |
|---|---|---|
| LLM called | When fitting: 1 code-generation conversation | When fitting: ⌈`number_of_rows_to_generate` / 100⌉ requests to the batch model |
| Optimisable | **yes** | no |
| Source | `sempipes/operators/sem_augment.py` (`CodeDataAugmentor`) | same file (`DirectDataAugmentor`) |

## Signature

```python
DataOp.sem_augment(
    nl_prompt: str,
    number_of_rows_to_generate: int,
    name: str,
    **kwargs,      # generate_via_code: bool = True
) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `nl_prompt` | `str` | required | What kind of rows to generate (e.g. "more examples of fraudulent baskets", "keep values realistic"). |
| `number_of_rows_to_generate` | `int` | required | How many rows to append. |
| `name` | `str` | required | Unique operator name (optimiser, hidden variables). Must not contain `__`. |
| `generate_via_code` | `bool` (keyword) | `True` | `True` selects code mode with SDV; `False` selects direct mode, where the LLM writes JSON rows. |

## How it works

### Code mode

At fit time, the code-generation model receives:

* 10 **randomly sampled rows** (dtype and missing percentage per column);
* the number of rows and the number of rows to generate;
* the installed SDV version and two example snippets (plain and conditional sampling);
* your `nl_prompt`;
* under the optimiser, also the dataset description (`as_X`), earlier attempts and inspirations.

The LLM must answer with `augment_data(df) -> df`, which returns the original rows plus the new ones. SemPipes validates it on the **first 1000 rows**: the columns must be the same, and there must be exactly `number_of_rows_to_generate` additional rows. Errors are fed back, up to `max_retries_for_code_gen` attempts. The accepted code is then run on the full training data.

**If every attempt fails**, SemPipes logs an error and falls back to an identity function, so **no rows are added**.

### Direct mode

For every chunk of up to 100 rows, the batch model receives `describe(include="all")` of the data, 10 random sample rows, the column list and your `nl_prompt`, and returns a JSON array of rows. The chunks are parsed with `pandas.read_json`, concatenated and appended. If the LLM returns more rows than requested, a random subset is kept. Fewer rows are **not** topped up. Unparsable JSON is retried up to 5 rounds. If a chunk still fails, the step raises.

## Example

Augment the **full table (features and target)** before splitting X and y, because new rows need labels too:

```python
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

import sempipes

rng = np.random.default_rng(0)
patients = pd.DataFrame(
    {
        "age": rng.integers(20, 80, size=300),
        "bmi": rng.normal(26, 4, size=300).round(1),
        "smoker": rng.choice(["yes", "no"], size=300, p=[0.2, 0.8]),
    }
)
patients["high_risk"] = ((patients["age"] > 65) & (patients["smoker"] == "yes")).astype(int)

data = skrub.var("patients", patients)
augmented = data.sem_augment(
    nl_prompt="Generate additional realistic high-risk patients (high_risk = 1).",
    number_of_rows_to_generate=100,
    name="more_high_risk",
)
print(augmented.skb.preview().shape)        # (400, 4)

X = augmented.drop(columns="high_risk", errors="ignore").skb.mark_as_X()
y = augmented["high_risk"].skb.mark_as_y()
pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(HistGradientBoostingClassifier(), y=y)
```

`errors="ignore"` lets the same pipeline predict on test data that has no `high_risk` column. During `predict`, `sem_augment` passes the rows through unchanged, so you get one prediction per test row.

## Pitfalls and tips

* **Needs at least 10 rows.** Both modes sample 10 example rows without replacement and raise a `ValueError` on smaller tables.
* **Augmentation sits before the X/y split, so cross-validation is optimistic.** skrub runs everything before `mark_as_X()`/`mark_as_y()` once on the full data and only then splits into folds. Synthetic rows (generated from all rows) end up in the validation folds too. Evaluate on a separate held-out test set with `learner.fit(train_env)` / `learner.predict(test_env)` (see [concepts](../concepts.md#traintest-discipline)).
* **Silent fallback in code mode.** If code generation fails, the data is returned without new rows. Check the shape.
* Direct mode can return slightly different dtypes or missing values (`pandas.read_json` infers types). Check and clean the new rows.
* Direct mode is limited by the LLM's output length and cost. Use code mode for hundreds or thousands of rows.

## See also

* Example script: [`examples/09_sem_augment.py`](../../examples/09_sem_augment.py)
* Notebook: [`examples/notebooks/demo_sem_augment.ipynb`](../../examples/notebooks/demo_sem_augment.ipynb)
* Related: [`sem_distill`](sem_distill.md) (the opposite: fewer rows), [optimizer](../optimizer.md).
