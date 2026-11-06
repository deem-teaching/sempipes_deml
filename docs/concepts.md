# Core concepts

This page covers what you need before using the individual operators:

1. [skrub DataOps in ten minutes](#skrub-dataops-in-ten-minutes)
2. [The two execution styles](#the-two-execution-styles)
3. [When is the LLM called?](#when-is-the-llm-called)
4. [Train/test discipline](#traintest-discipline)
5. [Optimisable operators and hidden variables](#optimisable-operators-and-hidden-variables)
6. [Privacy and safety](#privacy-and-safety)
7. [Files SemPipes writes](#files-sempipes-writes)
8. [Operator overview](#operator-overview)

## skrub DataOps in ten minutes

SemPipes does not have its own pipeline abstraction. It adds methods to [skrub](https://skrub-data.org/) **DataOps**, which are lazily recorded computations over dataframes. If you know pandas and scikit-learn, DataOps will feel familiar.

`import sempipes` monkey-patches the skrub classes. After the import, every `DataOp` has the methods `.sem_select`, `.sem_clean`, `.sem_refine`, `.sem_fillna`, `.sem_extract_features`, `.sem_gen_features`, `.sem_agg_features`, `.sem_augment` and `.sem_distill`, and the `.skb` namespace gains `.skb.apply_with_sem_choose`. The top-level package exports `sempipes.sem_choose`, `sempipes.as_X`, `sempipes.as_y`, `sempipes.get_config`, `sempipes.update_config` and `sempipes.LLM`.

### Variables, previews and evaluation

```python
import pandas as pd
import skrub
import sempipes

df = pd.DataFrame({"city": ["Berlin", "Paris"], "temp_c": [12.0, 15.5]})

data = skrub.var("weather", df)            # a named input; the value is optional
data = data.assign(temp_f=data["temp_c"] * 9 / 5 + 32)   # pandas code, recorded lazily
print(data)                               # shows a *preview* computed on df
print(data.skb.eval())                    # re-runs everything (fit_transform) and returns the result
print(data.skb.eval({"weather": df.head(1)}))  # ...or run it on other data
```

* `skrub.var(name, value)` declares an input. The **name** is how you pass data later. The **value** is optional example data.
* When a variable has a value, skrub computes a **preview immediately** every time you add a step. This catches errors early, but it also means that **SemPipes operators call the LLM while you are building the pipeline**. To get the cached preview result without recomputing anything, use `.skb.preview()`.
* `.skb.eval()` re-runs the whole graph from scratch (`fit_transform`), including the LLM calls.
* To make previews cheaper, `.skb.subsample(n=100)` limits them to a subsample of the data. `eval`, `fit` and cross-validation still use all rows. You can also create variables *without* a value (`skrub.var("weather")`). Then nothing runs until you provide data.

### Estimators, X and y

Assume that `data` is a DataOp over a table with a `label` column:

```python
from sklearn.ensemble import HistGradientBoostingClassifier

X = data.drop(columns="label").skb.mark_as_X()
y = data["label"].skb.mark_as_y()
pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(HistGradientBoostingClassifier(), y=y)
```

* `.skb.apply(estimator, y=...)` adds a scikit-learn transformer or predictor to the graph. SemPipes operators are scikit-learn-style estimators that are added in exactly this way. The two exceptions are `sem_select`, which uses `.skb.select(...)` with an LLM-backed selector, and `sem_choose`, which configures *your* estimator before applying it.
* `.skb.mark_as_X()` and `.skb.mark_as_y()` tell skrub where the features and the target are. Cross-validation splits the data **at these marks**, so everything before them is computed once on the full data (see [train/test discipline](#traintest-discipline)).
* `sempipes.as_X(op, description)` and `sempipes.as_y(op, description)` mark X/y and also attach a natural-language description. The descriptions go into the *pipeline summary* that the [optimiser](optimizer.md) hands to `sem_gen_features` (dataset and target description, model, task type, metric) and to `sem_augment` in code mode (dataset description). **Outside of `optimise_colopro` the descriptions are not sent to the LLM.** In that case, put important context into the `nl_prompt` itself.

```python
X = sempipes.as_X(skrub.var("responses"), "Survey answers of US residents about the Midwest")
y = sempipes.as_y(skrub.var("labels"), "The census region the respondent lives in")
```

### Learners: fit on train, predict on test

```python
learner = pred.skb.make_learner()     # an unfitted, sklearn-like object
learner.fit(env_train)                # env = {variable name: data}
predictions = learner.predict(env_test)
```

`make_learner(fitted=True)` fits on the variables' values right away. `learner.score(env)` scores the predictor with its default metric.

### Environments: always start from `get_data()`

An *environment* is a dict that maps variable names to data. Many SemPipes operators add **hidden skrub variables** to your graph (see [below](#optimisable-operators-and-hidden-variables)). When you pass an environment yourself, skrub does **not** fill in default values for variables you left out, and fails with:

```
UninitializedVariable: No value has been provided for 'sempipes_inspirations__my_features'
```

So build environments from `pipeline.skb.get_data()`, which contains the defaults of all variables that have a value, and add your own data on top:

```python
env = pred.skb.get_data()                 # includes the hidden sempipes_* variables
env.update({"weather": train_df})
learner.fit(env)
```

`predict`/`transform` do not need the hidden variables of the optimisable operators, but a graph that contains `sem_choose` needs its hidden variable in *every* environment. Building every environment from `get_data()` is the simplest rule.

### Splitting and cross-validation

```python
env = pred.skb.get_data()
env.update({"weather": df})

split = pred.skb.train_test_split(env, test_size=0.25, random_state=0)  # keeps all env keys
learner = pred.skb.make_learner()
learner.fit(split["train"])
print(learner.score(split["test"]))

results = pred.skb.cross_validate(env, cv=5, scoring="accuracy")           # pandas DataFrame
results = skrub.cross_validate(pred.skb.make_learner(), env, cv=5)          # same, functional form
```

Extra keyword arguments are forwarded to `sklearn.model_selection.train_test_split` and `sklearn.model_selection.cross_validate`. `split` also contains `split["X_test"]` and `split["y_test"]`.

### A complete example

```python
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

import sempipes

dataset = skrub.datasets.fetch_toxicity()            # tweets + "Toxic" / "Not Toxic" labels

tweets = sempipes.as_X(skrub.var("tweets"), "Short social-media posts (tweets)")
labels = sempipes.as_y(skrub.var("labels"), "Whether the tweet is toxic")

with_features = tweets.sem_extract_features(
    nl_prompt="Extract features that help to detect toxic language.",
    input_columns=["text"],
    name="tweet_features",
    output_columns={"insult_score": "How insulting the text is, from 0 (not at all) to 10 (very)."},
)
encoded = with_features.skb.apply(skrub.TableVectorizer())
pred = encoded.skb.apply(HistGradientBoostingClassifier(), y=labels)

env = pred.skb.get_data()
env.update({"tweets": dataset.X.head(300), "labels": dataset.y.head(300)})
split = pred.skb.train_test_split(env, test_size=0.25, random_state=0)

learner = pred.skb.make_learner()
learner.fit(split["train"])            # the LLM writes the extraction code, using training data only
print(learner.score(split["test"]))    # the same code is re-used on the test rows; no LLM call
```

## The two execution styles

### (a) Code generation: the LLM writes code once, at fit time

Most operators send the LLM a compact description of the **training** data, such as column names, dtypes, a handful of sample values and summary statistics (each operator page lists exactly what is sent). The LLM is asked for a Python function. SemPipes then:

1. Runs the function on a small sample of the data (between the first 50 and 1000 rows, depending on the operator) and checks the operator's contract: it must return a DataFrame, keep the row count, keep the original columns, and so on.
2. If the code raises or breaks the contract, sends the error back to the LLM and asks for a fix. This happens up to `max_retries_for_code_gen` times in total (default 5, see [configuration](configuration.md)).
3. Stores the accepted code in the fitted operator and **re-runs that code on every later `transform`**, for example on test data or new data, without asking the LLM again.

Generated code runs through `sempipes/code_generation/safe_exec.py` (see [Privacy and safety](#privacy-and-safety)). `sem_extract_features` runs it in a separate Python process so that broken GPU code cannot crash your session.

Code generation is cheap at prediction time and you can inspect the result. What happens when **all** attempts fail depends on the operator, so read the operator pages:

| Behaviour after the last failed attempt | Operators |
|---|---|
| Silently falls back to "do nothing" (identity) and logs an error | `sem_gen_features`, `sem_agg_features`, `sem_extract_features` (code mode), `sem_augment` (code mode) |
| Falls back to a simple model | `sem_fillna` (default mode): median / most-frequent value |
| Raises an error | `sem_clean`, `sem_refine`, `sem_distill` |
| Keeps the estimator's original parameter value | `sem_choose` |

### (b) Per-row processing: the LLM handles each row

Some operators (or modes) send **one LLM request per row**. Examples are filling a missing value from the rest of the row, or describing an image. The requests go through litellm's `batch_completion`, in chunks of `batch_size_for_batch_processing` concurrent requests (default 20), to the `llm_for_batch_processing` model. Per-row operators call the LLM **during every transform**, so also on test data and at prediction time. Cost and latency grow with the number of rows, and results are not cached.

| Operator | Code generation | Direct LLM processing (batch model) |
|---|---|---|
| `sem_fillna` | default: the LLM proposes imputation models | `with_llm_only=True` |
| `sem_extract_features` | default (`generate_via_code=True`): the LLM writes HuggingFace `transformers` code | `generate_via_code=False` |
| `sem_augment` | default (`generate_via_code=True`): the LLM writes SDV code | `generate_via_code=False`: the LLM writes the new rows itself as JSON (100 rows per request) |

## When is the LLM called?

A **fit** happens on every preview (when a variable has a value, i.e. while you build the graph), on every `.skb.eval()`, `learner.fit(...)` and `make_learner(fitted=True)`, **in every cross-validation fold**, and for every candidate of a hyperparameter search. A 5-fold cross-validation therefore generates code five times for each operator placed after `mark_as_X()`, and each fold may get different code. Operators placed before the X/y marks run once per cross-validation call.

| Operator | LLM calls when fitting (training data) | LLM calls when transforming new data |
|---|---|---|
| `sem_select` | 1 per column (sequential) | none: the selected column list is stored |
| `sem_clean`, `sem_refine`, `sem_gen_features`, `sem_agg_features`, `sem_distill` | 1 code-generation conversation | none: stored code is re-run |
| `sem_fillna` (default) | 1 code-generation conversation (5 candidate models) | none: the chosen model predicts |
| `sem_fillna(with_llm_only=True)` | 1 per missing value in the training data | 1 per missing value |
| `sem_extract_features` (code) | +1 if `output_columns=None` (feature proposal), then 1 code-generation conversation | none: stored code (and local models) re-run |
| `sem_extract_features` (per-row) | +1 if `output_columns=None`, then 1 per training row | 1 per row |
| `sem_augment` (code) / (direct) | 1 conversation / ⌈rows/100⌉ requests | none: data passes through unchanged |
| `sem_choose` | **eagerly, when you call `apply_with_sem_choose`**: 1 conversation per parameter, before any data is seen | none |

"Conversation" means one request plus up to `max_retries_for_code_gen - 1` retries.

The logs contain a line `TOKEN_USAGE,<input tokens>,<output tokens>` for every LLM response. Use these lines to estimate cost (see [configuration](configuration.md#logging)).

## Train/test discipline

* **Fit on training data only.** The LLM sees samples and statistics of the data it is fitted on. If you fit on the full dataset, information from the test set ends up in the prompt and therefore in the code.
* **Code is fixed, but statistics inside it are not.** A generated function is plain Python with no fitted state. If it computes a median, a quantile or a frequency *inside* the function, that statistic is recomputed on whatever data you pass at transform time, which could be the test set or a single row. Read the generated code (`verbose_code_synthesis=True`, see [configuration](configuration.md)). If needed, ask in the prompt for constants derived from the training data.
* **Per-row operators see test rows.** `sem_fillna(with_llm_only=True)` and per-row `sem_extract_features` send every test row to the LLM at prediction time.
* **`sem_augment` and `sem_distill` only change training data.** They act in `fit_transform` and pass data through unchanged in `transform`, so test data is never augmented or distilled. Because new rows need labels, apply them to the full table (features *and* target) *before* separating X and y.
* **Everything before `mark_as_X()`/`mark_as_y()` runs once on the full data during cross-validation and hyperparameter search.** skrub materialises X and y first and only then splits them into folds. A semantic operator placed before the marks is fitted on *all* rows, including the validation folds. This includes `sem_augment`/`sem_distill`, which have to sit there. For an honest estimate, keep a separate held-out test set and use `learner.fit(train)` / `learner.predict(test)`.
* **The optimiser only sees the data you give it.** `optimise_colopro` scores candidates with cross-validation on the variables' values plus `additional_env_variables`. Pass training data only (see [optimizer.md](optimizer.md)).

## Optimisable operators and hidden variables

Four operators can be improved by the [optimiser](optimizer.md). They write code iteratively and can replay earlier attempts:

* `sem_gen_features`
* `sem_agg_features`
* `sem_extract_features` with `generate_via_code=True` (the default)
* `sem_augment` with `generate_via_code=True` (the default)

Each of them takes a `name` argument and registers four hidden skrub variables:

| Variable | Default | Meaning |
|---|---|---|
| `sempipes_pipeline_summary__<name>` | `None` | Context about the pipeline (model, target, descriptions), filled in by the optimiser |
| `sempipes_prefitted_state__<name>` | `None` | If set to a state dict such as `{"generated_code": "..."}`, the operator **skips the LLM** and uses this code |
| `sempipes_memory__<name>` | `[]` | Earlier code versions with their scores, filled in by the optimiser |
| `sempipes_inspirations__<name>` | `[]` | High-scoring code from other branches of the search, filled in by the optimiser |

(`sem_extract_features` and `sem_augment` with `generate_via_code=False` create these variables but don't connect them to the graph, so the optimiser ignores them.)

Consequences:

* Names must be **unique** within a pipeline and must **not contain a double underscore** `__`. The optimiser recovers operator names by splitting the variable names at `__`.
* Build environments from `pipeline.skb.get_data()` (see [above](#environments-always-start-from-get_data)).
* You can reuse good code and skip the LLM entirely by setting the prefitted state. This is how you apply the optimiser's best result:

```python
env = pipeline.skb.get_data()
env["sempipes_prefitted_state__house_features"] = best_outcome.states.get("house_features")
```

## Privacy and safety

* **Data leaves your machine.** Column names, sample values, value counts and statistics are sent to the LLM provider. Per-row operators send whole rows, and images or audio files as base64. Only use data you are allowed to share with that provider.
* **`safe_exec` is not a security sandbox.** It runs generated code with a reduced set of builtins and an import allow-list (numpy, pandas, sklearn, skrub, scipy, torch, transformers, sdv, autogluon, xgboost, PIL, pathlib, urllib, huggingface_hub, …). `os`, `subprocess` and `sys` are blocked, but the allowed libraries can still read and write files (`pathlib`, `pandas`) and access the network (`urllib`, `huggingface_hub`). The allow-list protects against accidental mistakes, not against malicious code. Run SemPipes on your own machine and read the generated code before you rely on it.

## Files SemPipes writes

All paths are relative to the **current working directory**, so run scripts from the repository root.

| Path | Written by | Contents |
|---|---|---|
| `.sempipes_trajectories/<run_name>_<timestamp>_<id>.json` | `optimise_colopro`, at the end of a run | All trials, scores and generated code (see [optimizer.md](optimizer.md#trajectories)) |
| `.sem_gen_features_error_<timestamp>/` | `sem_gen_features`, when its stored code fails at transform time | `executed_code.py`, `stack_trace.txt` (the error is re-raised) |
| `.sem_agg_features_error_<timestamp>/` | `sem_agg_features`, same situation | `executed_code.py`, `stack_trace.txt` |
| `AutogluonModels/…` | AutoGluon, when the code generated by `sem_fillna` suggests a `TabularPredictor` | trained AutoGluon models |
| `~/.cache/huggingface/` | HuggingFace models downloaded by `sem_extract_features` code | model weights |

`sem_extract_features` also writes temporary pickle files into the system temp directory and deletes them afterwards. The error folders are safe to delete.

## Operator overview

| Operator | What it does | Style | Optimisable | Example |
|---|---|---|---|---|
| [`sem_select`](operators/sem_select.md) | Keep the columns that match a description | 1 small LLM decision per column | no | [01_sem_select.py](../examples/01_sem_select.py) |
| [`sem_clean`](operators/sem_clean.md) | Clean selected columns (types, whitespace, invalid values, …) | code generation | no | [02_sem_clean.py](../examples/02_sem_clean.py) |
| [`sem_refine`](operators/sem_refine.md) | Normalise and deduplicate the values of one column (typos, synonyms, languages) | code generation | no | [03_sem_refine.py](../examples/03_sem_refine.py) |
| [`sem_fillna`](operators/sem_fillna.md) | Impute missing values in one column | both: LLM-proposed models (default) / per-row | no | [04_sem_fillna.py](../examples/04_sem_fillna.py) |
| [`sem_extract_features`](operators/sem_extract_features.md) | Derive features from text, image or audio columns | both: generated `transformers` code (default) / per-row | yes (code mode) | [05_sem_extract_features_text.py](../examples/05_sem_extract_features_text.py), [06_sem_extract_features_images.py](../examples/06_sem_extract_features_images.py) |
| [`sem_gen_features`](operators/sem_gen_features.md) | Add new feature columns computed from existing ones (CAAFE-style) | code generation | yes | [07_sem_gen_features.py](../examples/07_sem_gen_features.py) |
| [`sem_agg_features`](operators/sem_agg_features.md) | Join a second table and aggregate it into features | code generation | yes | [08_sem_agg_features.py](../examples/08_sem_agg_features.py) |
| [`sem_augment`](operators/sem_augment.md) | Add synthetic training rows | both: SDV code (default) / LLM-written rows | yes (code mode) | [09_sem_augment.py](../examples/09_sem_augment.py) |
| [`sem_distill`](operators/sem_distill.md) | Shrink the training data to fewer rows | code generation | no | [10_sem_distill.py](../examples/10_sem_distill.py) |
| [`sem_choose`](operators/sem_choose.md) | Let the LLM propose hyperparameter search spaces | code generation (once, when the graph is built) | no (tuned by hyperparameter search) | [11_sem_choose.py](../examples/11_sem_choose.py) |
