# `sem_refine`: normalise and deduplicate the values of one column

`sem_refine` cleans up a messy categorical column. Examples are country codes written in many ways (`"USA "`, `"u-s"`, `"US"`), product types with synonyms (`"soccer ball"`, `"football"`), or city names with typos and translations (`"Roma"`, `"Rome, Italy"`). The LLM looks at **all distinct values with their counts** and writes a function that maps variants to canonical values, typically via a correspondence dictionary. Outliers can be mapped to `"unknown"`.

| | |
|---|---|
| Style | Code generation |
| LLM called | When fitting: 1 code-generation conversation. Never at transform time. |
| Optimisable | no |
| Source | `sempipes/operators/sem_refine.py` |

## Signature

```python
DataOp.sem_refine(target_column: str, nl_prompt: str, refine_with_existing_values_only: bool) -> DataOp
```

All three arguments are required.

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `target_column` | `str` | required | The column to refine. |
| `nl_prompt` | `str` | required | What the canonical values should look like, e.g. "ISO 3166-1 alpha-2 uppercase codes" or "group similar sports items". |
| `refine_with_existing_values_only` | `bool` | required | `True` tells the LLM not to introduce new unique values. `False` allows new values (e.g. a new umbrella category). This is **only an instruction in the prompt**; nothing checks it. |

## How it works

At **fit** time, the code-generation model receives:

* the column name and dtype;
* `df[target_column].value_counts()` as JSON, i.e. **every distinct value with its count**;
* the existing-values instruction and your `nl_prompt`;
* three worked examples that map variants to canonical values with a dictionary.

The LLM must answer with `refine_column(df) -> pandas.DataFrame`, which returns the whole table with the target column refined. SemPipes runs it on the **first 50 rows**, checks that the row count is unchanged, runs a NaN sanity check (see the known issue below) and retries with the error message if needed.

At **transform** time, the stored function is applied to the full table.

**If every attempt fails**, the step **raises** `KeyError: 'refine_column'`. There is no fallback.

## Example

```python
import pandas as pd
import skrub

import sempipes

countries = pd.DataFrame(
    {"country_code": ["US", "usa", "USA ", " US", "U-S", "uk", "UK ", "GB", "FR", "fra", "FRN", "DE", "deu", "GER", "Hamburg"]}
)
data = skrub.var("countries", countries)

refined = data.sem_refine(
    target_column="country_code",
    nl_prompt="Make sure that all values are in the ISO 3166-1 alpha-2 two-letter uppercase format",
    refine_with_existing_values_only=False,
)
print(refined.skb.preview()["country_code"].value_counts())
```

## See also

* Example script: [`examples/03_sem_refine.py`](../../examples/03_sem_refine.py)
* Related: [`sem_clean`](sem_clean.md) for general multi-column cleaning, [`sem_fillna`](sem_fillna.md) for missing values.
