# `sem_select`: keep the columns that match a description

`sem_select` keeps the columns of a table that match a natural-language description, such as "columns measured in the metric system" or "columns that could help predict the price". For each column, the LLM decides "keep" or "drop" based on the column's name, a few sample values and some simple statistics. The selected column list is stored when the operator is fitted, so test data gets exactly the same columns.

| | |
|---|---|
| Style | One small LLM decision per column (uses the code-generation model) |
| LLM called | When fitting: 1 request per column, sent one after another. Never at transform time. |
| Optimisable | no |
| Source | `sempipes/operators/sem_select_llm.py` |

## Signature

```python
DataOp.sem_select(nl_prompt: str) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `nl_prompt` | `str` | required | Describes which columns to **keep**. The text is inserted into each per-column prompt as "the data scientist wants you to take special care of the following". |

## How it works

`sem_select` builds a skrub selector (`skrub.selectors.Filter`) and applies it with `.skb.select(...)`. When the operator is fitted, the filter runs once per column. For each column, SemPipes sends the **code-generation model** the following:

* the column name;
* up to 10 randomly sampled values (no fixed seed, so the prompt can differ between runs);
* the number of unique values, whether the column has nulls or is all null, and whether it is sorted;
* type flags (bool, numeric, integer, float, string, object, date, duration, categorical);
* for numeric columns: minimum, maximum and mean;
* your `nl_prompt`.

The LLM answers with a one-line Python snippet `__column_is_relevant = True` (or `False`), which is executed with `safe_exec`. skrub stores the list of selected columns at fit time and selects the same list at transform time, without new LLM calls.

**Failure handling:** if anything goes wrong for a column (network error, invalid API key, unparsable answer), SemPipes prints `An error occurred for column <name>: ...` and treats the column as **not relevant**. The column is dropped and no exception is raised.

## Example

```python
import pandas as pd
import skrub

import sempipes  # adds the .sem_* methods to skrub DataOps

df = pd.DataFrame(
    {
        "height_mm": [297.0, 420.0],
        "width_mm": [210.0, 297.0],
        "kind": ["A4", "A3"],
        "ID": [4, 3],
    }
)
some_df = skrub.var("some_df", df)

metric = some_df.sem_select("Columns whose values are in the metric system")
print(metric.skb.preview())   # most likely height_mm and width_mm
```

## See also

* Example script: [`examples/01_sem_select.py`](../../examples/01_sem_select.py)
* Notebook: [`examples/notebooks/demo__sem_select.ipynb`](../../examples/notebooks/demo__sem_select.ipynb)
* Related: [`sem_clean`](sem_clean.md) for cleaning the columns you keep. See [concepts](../concepts.md) for previews and when the LLM is called.
