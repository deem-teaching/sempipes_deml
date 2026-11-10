# `sem_clean`: clean selected columns with LLM-written code

`sem_clean` asks the LLM to write a cleaning function for a list of columns, following your instructions. Typical tasks are stripping whitespace, fixing capitalisation, coercing strings to numbers, treating `"NA"`/`"n/a"` as missing, or handling implausible values. The function is written once from the training data and then re-applied to any new data.

| | |
|---|---|
| Style | Code generation |
| LLM called | When fitting: 1 code-generation conversation (up to `max_retries_for_code_gen` attempts). Never at transform time. |
| Optimisable | no |
| Source | `sempipes/operators/sem_clean.py` |

## Signature

```python
DataOp.sem_clean(nl_prompt: str, columns: list[str]) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `nl_prompt` | `str` | required | What to clean and how. Be specific (e.g. "coerce age to integers, treat 'NA' as missing"). |
| `columns` | `list[str]` | required | The columns to clean. Other columns are passed through untouched. |

## How it works

At **fit** time, the code-generation model receives:

* the column names;
* for each column: its dtype, the percentage of missing values and its **first 10 values**;
* `df[columns].describe(include="all")`;
* the number of rows, and the instruction not to change it;
* your `nl_prompt`;
* a hint to use `BayesianRidge`-style estimators if it uses scikit-learn's `IterativeImputer`.

The LLM must answer with a function `clean_columns(df, columns) -> pandas.DataFrame`. SemPipes runs it on the **first 50 rows** and checks that it returns a DataFrame with the same number of rows. If that fails, the error goes back to the LLM for another attempt, up to `max_retries_for_code_gen` attempts in total.

At **transform** time, the stored function is applied to `df[columns]` and the result is written back into those columns. The rest of the table is unchanged.

**If every attempt fails**, no code is stored and the step **raises** (`KeyError: 'clean_columns'`) as soon as it is applied. There is no silent fallback.

## Example

```python
import pandas as pd
import skrub

import sempipes

df = pd.DataFrame(
    {
        "name": [" Alice ", "bob", None, "Charlie  ", "Martina"],
        "age": ["25", "30 ", "NA", "40", "150"],
    }
)
data = skrub.var("people", df)

cleaned = data.sem_clean(
    nl_prompt=(
        "Strip whitespace from names and capitalise them. "
        "Coerce age to integers; treat 'NA' and ages above 120 as missing."
    ),
    columns=["name", "age"],
)
print(cleaned.skb.preview())
```


## See also

* Example script: [`examples/02_sem_clean.py`](../../examples/02_sem_clean.py)
* Related: [`sem_refine`](sem_refine.md) to normalise the *values* of a categorical column (synonyms, typos, languages) and [`sem_fillna`](sem_fillna.md) for imputation. See also [concepts → train/test discipline](../concepts.md#traintest-discipline).
