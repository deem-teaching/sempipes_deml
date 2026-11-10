# `sem_agg_features`: join and aggregate a second table into features

Often the information you need sits in a **second table with many rows per entity**, for example orders per customer or products per shopping basket. `sem_agg_features` asks the LLM to write a function that aggregates the right-hand table per key (counts, sums, means, shares, …) and **left-joins** the result onto your main table. The output has exactly one row per row of the main table.

| | |
|---|---|
| Style | Code generation |
| LLM called | When fitting: 1 code-generation conversation. Never at transform time. |
| Optimisable | **yes** (see [optimizer](../optimizer.md)) |
| Source | `sempipes/operators/sem_agg_features.py` |

## Signature

```python
DataOp.sem_agg_features(
    right_data_op: DataOp,
    left_on: str,
    right_on: str,
    nl_prompt: str,
    name: str,
    how_many: int = 10,
) -> DataOp
```

Call it on the **left** (main) table.

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `right_data_op` | `DataOp` | required | The table to aggregate, e.g. `skrub.var("orders", orders_df)`. |
| `left_on` | `str` | required | Join key in the left table. |
| `right_on` | `str` | required | Join key in the right table. Any column with this name is dropped from the result, so if `left_on == right_on`, the left key disappears too (see pitfalls). |
| `nl_prompt` | `str` | required | What the features should capture. |
| `name` | `str` | required | Unique operator name (optimiser, hidden variables). Must not contain `__`. |
| `how_many` | `int` | `10` | Number of new features requested in the prompt. It is not enforced. |

## How it works

At **fit** time, the code-generation model receives:

* for **both** tables and every column: dtype, missing ratio and up to 10 random non-missing values (`random_state=42`);
* the two join keys;
* the full `describe(include="all")` of the right table (the LLM may use these statistics as constants);
* your `nl_prompt` and `how_many`;
* during optimisation, also memory and inspiration code (see [optimizer](../optimizer.md)).

The LLM must answer with `_sem_agg_join(left_join_column, left_df, right_join_column, right_df)`. SemPipes validates it on the **first 100 left rows** (together with the right rows for 90% of their keys). The result must have the same number of rows as the left input and keep every left column. Pandas `_x` suffixes on left columns are undone automatically. Errors are fed back, up to `max_retries_for_code_gen` attempts.

At **transform** time, the stored function runs on copies of both tables. The right join key is dropped, and SemPipes asserts that the row count is unchanged. If the code raises, the code and the stack trace are written to `.sem_agg_features_error_<timestamp>/` and the error is re-raised.

**If every attempt fails**, SemPipes logs an error and falls back to returning the left table unchanged (**no new features**).

## Example

```python
import pandas as pd
import skrub

import sempipes

customers = pd.DataFrame({"customer_id": [1, 2, 3, 4], "country": ["DE", "FR", "DE", "IT"]})
orders = pd.DataFrame(
    {
        "customer_id": [1, 1, 2, 3, 3, 3, 4],
        "amount": [20.0, 35.5, 12.0, 80.0, 5.0, 42.0, 18.0],
        "category": ["books", "toys", "books", "electronics", "books", "toys", "books"],
    }
)

customers_op = skrub.var("customers", customers)
orders_op = skrub.var("orders", orders)

enriched = customers_op.sem_agg_features(
    orders_op,
    left_on="customer_id",
    right_on="customer_id",
    nl_prompt="Summarise each customer's spending behaviour and preferred categories.",
    name="order_features",
    how_many=5,
)
print(enriched.skb.preview())   # country + new features; customer_id is dropped (same name on both sides)
```

To predict on new data, the environment must contain **both** tables, e.g. `{"customers": new_customers, "orders": all_orders}`. Build it from `pipeline.skb.get_data()` (see [concepts](../concepts.md#environments-always-start-from-get_data)).


## See also

* Example script: [`examples/08_sem_agg_features.py`](../../examples/08_sem_agg_features.py)
* Notebook: [`examples/notebooks/demo__sem_agg_features.ipynb`](../../examples/notebooks/demo__sem_agg_features.ipynb)
* Related: [`sem_gen_features`](sem_gen_features.md) (features from the same table), [optimizer](../optimizer.md).
