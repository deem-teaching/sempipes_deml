# SemPipes documentation

Read the pages in this order. If you only have 30 minutes, read steps 1–3 and the page of the operator you need.

| # | Page | What you learn |
|---|---|---|
| 1 | [Installation](installation.md) | Python 3.11/3.12, Poetry, API keys, verifying the setup, troubleshooting |
| 2 | [Core concepts](concepts.md) | skrub DataOps crash course, the two execution styles, when the LLM is called, train/test discipline, hidden variables, files written |
| 3 | [Configuration](configuration.md) | Choosing models and providers (Gemini, OpenAI, Ollama), batch size and rate limits, retries, logging |
| 4 | Operators (below) | One page per operator: signature, arguments, what the LLM sees, pitfalls, runnable snippet |
| 5 | [Optimizer](optimizer.md) | Improving generated code automatically with `optimise_colopro`, applying the best result, trajectories, cost |

## Operators

Pages are grouped by task. Each one links to its example script in [`examples/`](../examples/README.md).

| Task | Operator | One-liner |
|---|---|---|
| Select | [`sem_select`](operators/sem_select.md) | Keep the columns that match a description |
| Clean | [`sem_clean`](operators/sem_clean.md) | Clean selected columns with generated code |
| | [`sem_refine`](operators/sem_refine.md) | Normalise and deduplicate the values of one column |
| | [`sem_fillna`](operators/sem_fillna.md) | Impute missing values (LLM-chosen model or per-row LLM) |
| Feature engineering | [`sem_extract_features`](operators/sem_extract_features.md) | Features from text, images and audio |
| | [`sem_gen_features`](operators/sem_gen_features.md) | New columns computed from existing ones (CAAFE-style) |
| | [`sem_agg_features`](operators/sem_agg_features.md) | Aggregate a second table and join it |
| Training data | [`sem_augment`](operators/sem_augment.md) | Add synthetic training rows |
| | [`sem_distill`](operators/sem_distill.md) | Reduce the training data to fewer rows |
| Model selection | [`sem_choose`](operators/sem_choose.md) | LLM-proposed hyperparameter search spaces |

An overview table with execution style, optimisability and example scripts is in [concepts.md](concepts.md#operator-overview).

## The five things that trip people up most

1. **Building a pipeline already calls the LLM** when your `skrub.var(...)` has a value, because skrub computes previews eagerly. See [concepts](concepts.md#variables-previews-and-evaluation).
2. **Environments must come from `pipeline.skb.get_data()`**, otherwise you get `UninitializedVariable: ... 'sempipes_...'`. See [concepts](concepts.md#environments-always-start-from-get_data).
3. **Some operators fail silently.** If code generation fails, `sem_gen_features`, `sem_agg_features`, `sem_extract_features` (code mode) and `sem_augment` (code mode) add nothing, and `sem_select` drops columns whose LLM call failed. Always look at the output.
4. **Per-row operators call the LLM on test data too**, every time. See [concepts](concepts.md#when-is-the-llm-called).
5. **`update_config` does not reach new threads or worker processes** (`n_jobs`). See [configuration](configuration.md#caveat-the-config-is-per-thread--per-context).

## More material

* [`examples/README.md`](../examples/README.md): runnable scripts, one per operator, plus an optimiser example. Run them from the repository root, e.g. `poetry run python examples/01_sem_select.py`.
* [`examples/notebooks/`](../examples/notebooks/): Jupyter notebooks with larger demos (start with `poetry run jupyter lab`).
* skrub's own documentation on DataOps: <https://skrub-data.org/>.
