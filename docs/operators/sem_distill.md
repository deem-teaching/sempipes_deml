# `sem_distill`: shrink the training data

`sem_distill` asks the LLM to write code that reduces the training data to `number_of_rows` representative rows. It may use stratified subsampling, clustering/coreset selection or prototype selection, as described in your prompt. This is useful when training is expensive or the data is highly redundant. Like [`sem_augment`](sem_augment.md), it only acts when fitting; at transform time the data passes through unchanged.

| | |
|---|---|
| Style | Code generation |
| LLM called | When fitting: 1 code-generation conversation |
| Optimisable | no |
| Source | `sempipes/operators/sem_distill.py` |

## Signature

```python
DataOp.sem_distill(nl_prompt: str, number_of_rows: int) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `nl_prompt` | `str` | required | How to distill, e.g. "keep the class balance of `label`" or "keep rare categories". |
| `number_of_rows` | `int` | required | Target number of rows of the distilled training data. |

## How it works

At fit time, the code-generation model receives:

* `describe(include="all")` of the data;
* up to 10 random sample rows (dtype and missing percentage per column);
* the number of rows and your `nl_prompt`;
* suggestions of techniques (subsampling, SMOTE, gradient matching, coresets, …), and the information that it may use `torch`/`transformers`.

The LLM must answer with `distill_data(df, number_of_rows) -> df`. SemPipes validates it by distilling the **first 100 rows down to 50**: the columns must be unchanged and exactly 50 rows must come back. Errors are fed back, up to `max_retries_for_code_gen` attempts. The accepted code is then run on the full training data with your `number_of_rows`.

**If every attempt fails**, the step **raises** `KeyError: 'distill_data'`. There is no fallback.

## Example

As with `sem_augment`, distill the **full table (features and target)** before splitting X and y, so that labels stay aligned:

```python
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

import sempipes

rng = np.random.default_rng(0)
customers = pd.DataFrame(
    {
        "age": rng.integers(18, 90, size=2000),
        "income": rng.normal(45_000, 15_000, size=2000).round(-2),
        "segment": rng.choice(["student", "professional", "retired"], size=2000),
    }
)
customers["churned"] = (rng.random(2000) < 0.1).astype(int)

data = skrub.var("customers", customers)
distilled = data.sem_distill(
    nl_prompt="Keep the class balance of 'churned' and cover all segments.",
    number_of_rows=200,
)
print(distilled.skb.preview().shape)    # (200, 4)

X = distilled.drop(columns="churned", errors="ignore").skb.mark_as_X()
y = distilled["churned"].skb.mark_as_y()
pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(HistGradientBoostingClassifier(), y=y)
```

## Pitfalls and tips

* **Needs at least ~50 rows.** The validation step asks for 50 rows from the first 100. With fewer than 50 training rows, every attempt fails and the step raises.
* **Raises on failure.** Unlike the optimisable operators, there is no silent fallback.
* **The final size is not checked.** Only the validation run (100 → 50 rows) is checked. Verify the shape of the distilled data yourself.
* **Cross-validation sees the distilled data only.** Distillation sits before the X/y split, so skrub runs it once on the full data and then cross-validates on the distilled rows. Evaluate on a separate test set (see [concepts](../concepts.md#traintest-discipline)).
* The prompt mentions libraries that are not on the [`safe_exec`](../concepts.md#privacy-and-safety) allow-list or not installed (e.g. `torchvision`, SMOTE implementations). Code that imports them fails and is retried. Steer the LLM with your prompt ("use scikit-learn / pandas only") if you see such errors.

## See also

* Example script: [`examples/10_sem_distill.py`](../../examples/10_sem_distill.py)
* Related: [`sem_augment`](sem_augment.md) (more rows instead of fewer), [concepts → train/test discipline](../concepts.md#traintest-discipline).
