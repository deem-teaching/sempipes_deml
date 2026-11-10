# `sem_choose`: let the LLM propose hyperparameter search spaces

`sem_choose` replaces hand-written hyperparameter grids with descriptions. Instead of typing `max_depth=skrub.choose_from([3, 5, 8, None])`, you write `max_depth="Common range of values for the maximum depth of the trees"`. The LLM translates each description into a `skrub.choose_from([...])` over concrete values. You then tune the pipeline with skrub's usual hyperparameter search.

It consists of two pieces:

* `sempipes.sem_choose(name, **param_to_nl_prompt)` records your requests. No LLM call happens here.
* `DataOp.skb.apply_with_sem_choose(estimator, choices, ...)` asks the LLM for the choices and applies the estimator, like `.skb.apply`.

| | |
|---|---|
| Style | Code generation |
| LLM called | **Eagerly, when `apply_with_sem_choose` is called** (graph construction): 1 conversation per parameter. Never during fit or transform. |
| LLM sees data? | No: only the estimator class, parameter names, your descriptions and default values |
| Optimisable | no. The proposed choices are tuned by hyperparameter search (also inside the [optimizer](../optimizer.md)). |
| Source | `sempipes/operators/sem_choose_llm.py` |

## Signatures

```python
sempipes.sem_choose(name: str, **kwargs: str) -> SemChoices

DataOp.skb.apply_with_sem_choose(
    estimator,                       # a scikit-learn estimator instance
    choices: SemChoices,
    *,
    y=None,
    cols=skrub.selectors.all(),
    exclude_cols=None,
    how: str = "auto",
    allow_reject: bool = False,
    unsupervised: bool = False,
) -> DataOp
```

## Arguments

`sem_choose`:

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of this set of choices. It is used to name the skrub choices (`__sempipes__<name>__<param>`) and a hidden variable. |
| `**kwargs` | `str` | — | One entry per estimator parameter: `parameter_name="description of the values to try"`. |

`apply_with_sem_choose`:

| Name | Type | Default | Description |
|---|---|---|---|
| `estimator` | estimator | required | The estimator whose parameters are chosen. **It is modified in place** via `set_params`. |
| `choices` | `SemChoices` | required | The result of `sempipes.sem_choose(...)`. |
| `y`, `cols`, `exclude_cols`, `how`, `allow_reject`, `unsupervised` | | as in `.skb.apply` | Passed unchanged to skrub's `.skb.apply`. |

## How it works

For each `parameter=description` pair, `apply_with_sem_choose` immediately sends the code-generation model:

* the estimator's class path (e.g. `sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier`) and the parameter name;
* your description;
* the parameter's default value, if the estimator defines one ("may be a reasonable choice to include");
* the names and versions of **all installed Python packages**, which makes the prompt long;
* two worked examples, and on retries the tracebacks of earlier failures.

The LLM answers with `__generated_sempipes_choices = skrub.choose_from([...])`. SemPipes executes it, names the choice `__sempipes__<name>__<param>` and sets it on the estimator. The LLM is told that the **first value is the most important**, because skrub uses the first option of `choose_from` as the default when you don't run a search.

**If every attempt for a parameter fails**, the parameter keeps its original value. Only the errors are printed.

Finally, the estimator is applied with `.skb.apply(...)`. The graph also gets a hidden variable `sempipes__choices__<name>__choices`.

## Example

```python
import skrub
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier

import sempipes
from sempipes import sem_choose

data = load_breast_cancer(as_frame=True)
X = skrub.var("X", data.data).skb.mark_as_X()
y = skrub.var("y", data.target).skb.mark_as_y()

model = X.skb.apply_with_sem_choose(           # <- the LLM is called here, once per parameter
    HistGradientBoostingClassifier(),
    y=y,
    choices=sem_choose(
        name="hgb",
        learning_rate="A promising set of learning rates to try",
        max_depth="Common range of values for the maximum depth of the learned trees",
    ),
)

search = model.skb.make_randomized_search(n_iter=10, cv=3, random_state=0, fitted=True)
print(search.results_)                         # one row per tried combination
best_learner = search.best_learner_
```

To use `best_learner` on new data, pass an environment built from `model.skb.get_data()`. The hidden `sempipes__choices__hgb__choices` variable is needed for **prediction too**, not only for fitting:

```python
env = model.skb.get_data()
env["X"] = new_X
predictions = best_learner.predict(env)
```

## See also

* Example script: [`examples/11_sem_choose.py`](../../examples/11_sem_choose.py)
* Notebook: [`examples/notebooks/demo.ipynb`](../../examples/notebooks/demo.ipynb)
* Related: skrub's [hyperparameter tuning guide](https://skrub-data.org/stable/), [optimizer](../optimizer.md).
