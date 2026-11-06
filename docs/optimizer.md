# Optimising semantic operators with `optimise_colopro`

A semantic operator's first attempt is rarely its best one. `optimise_colopro` (in `sempipes/optimisers/colopro.py`) improves the code of the [optimisable operators](concepts.md#optimisable-operators-and-hidden-variables) in a pipeline by **iterating with the LLM, using the pipeline's cross-validation score as feedback**. The idea follows OPRO ("optimisation by prompting"), applied operator by operator.

Optimisable operators are `sem_gen_features`, `sem_agg_features`, `sem_extract_features` (code mode) and `sem_augment` (code mode).

## How it works

1. **Find the operators.** Every node that has a hidden `sempipes_prefitted_state__<name>` variable is an optimisable operator. The operators are ordered topologically, upstream first.
2. **Summarise the pipeline (no LLM call).** The summary contains the final model's class, classification vs. regression (from `ClassifierMixin`/`RegressorMixin`), the target's name and `as_y` description, the dataset's `as_X` description and your `scoring`. `sem_gen_features` puts all of this into its prompt, `sem_augment` uses the dataset description, and all optimisable operators use the metric name when reporting earlier scores.
3. **Trial 0 (the root).** Each operator generates code once, in topological order, on the training data. The pipeline is then scored by **cross-validation**, with all generated code **frozen** through the prefitted state. No LLM calls happen inside the folds for these operators.
4. **Trials 1 … `num_trials − 1`.**
   * The **operator selection strategy** picks which operator to evolve.
   * The **search policy** picks which earlier trial (the *parent*) to start from. All other operators keep the parent's code.
   * The chosen operator is asked for new code. Its prompt now includes its **memory**: the code versions along the path from the root to the parent, each with its score and a note whether it improved things ("You may choose to copy from this previous version…"). Some operators also see **inspirations**, i.e. top-scoring code from other branches.
   * If the chosen operator changes its output columns (`sem_gen_features`, `sem_agg_features`, `sem_extract_features` without `output_columns`), the optimisable operators downstream of it are re-generated in the same trial.
   * The pipeline is cross-validated again. The result is recorded as a new node in the **search tree**.
5. All trials are returned as a list of `Outcome`s and saved as a [trajectory](#trajectories).

If the pipeline contains hyperparameter choices (`skrub.choose_*` or [`sem_choose`](operators/sem_choose.md)), each trial runs a **randomized search** (`num_hpo_iterations_per_trial` candidates × `cv` folds) instead of plain cross-validation. The trial's score is the best mean test score of that search.

## Signature

```python
from sempipes.optimisers import optimise_colopro

optimise_colopro(
    dag_sink: DataOp,
    num_trials: int,
    search: SearchPolicy = TreeSearch(),
    scoring: str = "accuracy",
    cv=5,
    num_hpo_iterations_per_trial: int = 10,
    pipeline_definition: Callable[..., DataOp] | None = None,
    run_name: str | None = None,
    additional_env_variables: dict[str, Any] | None = None,
    n_jobs_for_evaluation: int = -1,
    only_optimize_operator: str | None = None,
    optimize_all_operators: bool = False,
    operator_selection_strategy: str = "ucb",
    ucb_alpha: float = 0.2,
    ucb_c: float = 1.0,
    epsilon: float = 0.1,
) -> list[Outcome]
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `dag_sink` | `DataOp` | required | The last node of your pipeline, usually the predictor (`X.skb.apply(model, y=y)`). It must contain at least one optimisable operator. |
| `num_trials` | `int` | required | Number of trials, including trial 0, which generates code for **every** optimisable operator. |
| `search` | `SearchPolicy` | `TreeSearch()` | How to choose the parent of the next trial (see [search policies](#search-policies)). **Always pass a fresh object** (see [pitfalls](#pitfalls)). |
| `scoring` | `str` or callable | `"accuracy"` | scikit-learn scoring, where **higher must be better** (use e.g. `"neg_root_mean_squared_error"` for errors). The value is also shown to the LLM as the target metric, so prefer a string. |
| `cv` | int or CV splitter | `5` | Cross-validation used to score each trial. |
| `num_hpo_iterations_per_trial` | `int` | `10` | `n_iter` of the randomized search per trial. Only used if the pipeline contains choices. |
| `pipeline_definition` | callable or `None` | `None` | A function that builds the pipeline. Its source code is stored in the pipeline summary but is **not currently used in any prompt**. |
| `run_name` | `str` or `None` | `None` | File-name prefix of the saved trajectory (default `colopro`). |
| `additional_env_variables` | `dict` or `None` | `None` | Added to `dag_sink.skb.get_data()` for every evolution and scoring step. **Pass your training data here** (e.g. `{"responses": X_train, "labels": y_train}`). |
| `n_jobs_for_evaluation` | `int` | `-1` | `n_jobs` of the cross-validation when the pipeline has no choices. The randomized search (pipelines with choices) always uses `n_jobs=-1`. |
| `only_optimize_operator` | `str` or `None` | `None` | Name of one operator to focus on. Trial 0 and a warm-up (one trial per operator) still touch every operator; afterwards only this one is evolved. Overrides `operator_selection_strategy`. |
| `optimize_all_operators` | `bool` | `False` | Evolve **every** operator in every trial. Overrides both options above. |
| `operator_selection_strategy` | `str` | `"ucb"` | `"ucb"`, `"epsilon_greedy"`, `"adaevolve_global_norm"` or `"round_robin"` (see [below](#operator-selection-strategies)). Any other value raises `ValueError`. |
| `ucb_alpha` | `float` | `0.2` | Exploration parameter for `"ucb"`. |
| `ucb_c` | `float` | `1.0` | Exploration parameter for `"adaevolve_global_norm"`. |
| `epsilon` | `float` | `0.1` | Exploration rate for `"epsilon_greedy"` (must be in [0, 1]). |

**Returns** a list with one `Outcome` per trial, in trial order. The same list is stored in the search object.

**Which data is used?** Every step uses `dag_sink.skb.get_data()`, i.e. the values of all variables that have one, updated with `additional_env_variables`. The easiest setup is to create your data variables **without** values (`skrub.var("responses")`) and pass the training data via `additional_env_variables`. That way no LLM calls happen while you build the pipeline, and test data can't leak in.

## Search policies

Import them from `sempipes.optimisers`:

| Policy | Constructor | Behaviour |
|---|---|---|
| `TreeSearch` | `TreeSearch(min_num_drafts=2)` | Greedy. Keeps creating new *drafts* (children of the root, with better-than-root code from other branches as inspiration) until `min_num_drafts` drafts exist that have not been expanded yet. Otherwise expands the best-scoring trial so far. |
| `MonteCarloTreeSearch` | `MonteCarloTreeSearch(c=1.41, root_children=3, max_non_root_children=2)` | UCT on min–max normalised scores. The root gets up to `root_children` children and other nodes up to `max_non_root_children` before the search descends to the child with the highest UCT value. `c` controls exploration. |
| `EvolutionarySearch` | `EvolutionarySearch(population_size)` (required) | Picks the parent at random among the `population_size` best trials, with probability proportional to the score, so it assumes positive scores. |

## Operator selection strategies

These only matter when the pipeline has **two or more** optimisable operators. Every run starts with trial 0, in which every operator is generated. The strategy decides what happens from trial 1 on.

| Strategy | Behaviour |
|---|---|
| `"ucb"` (default) | Warm-up: one trial per operator. After that, a UCB1 bandit (via `mabwiser`) whose reward is the score improvement of a trial over its parent. `ucb_alpha` controls exploration. |
| `"epsilon_greedy"` | Same warm-up, then an ε-greedy bandit with exploration rate `epsilon`. |
| `"adaevolve_global_norm"` | Each operator is tried once. After that, UCB over decayed (ρ = 0.7), globally normalised improvements. `ucb_c` controls exploration. |
| `"round_robin"` | Cycles through the operators in topological order. |
| `optimize_all_operators=True` | Every operator in every trial. |
| `only_optimize_operator="name"` | After the warm-up, always that operator. |

## `Outcome`

`Outcome` is defined in `sempipes/optimisers/search_policy.py`:

| Field | Type | Meaning |
|---|---|---|
| `score` | `float` | Mean cross-validation score of this trial (or the best mean test score of its randomized search). Computed **on the training data**, so it is not a test score. |
| `states` | `OperatorStates` | The code of **every** optimisable operator in this trial. `outcome.states.get("my_op")` returns the state dict, e.g. `{"generated_code": "..."}`. For `sem_extract_features` without `output_columns` it also contains `"generated_features"`. `outcome.states.operators()` lists the names. |
| `search_node` | `SearchNode` | Position in the search tree: `.trial`, `.parent_trial`, `.operator_to_evolve`, `.operators_evolved`, `.parent_score`, `.memories`, `.fixed_states`, `.inspirations`. |
| `memory_updates` | `dict[str, str]` | The new code of each operator evolved in this trial. It becomes part of the children's memory. |

There is **no helper that applies the best outcome**. Copy the state into the environment yourself by setting `env["sempipes_prefitted_state__<name>"] = best.states.get("<name>")` before fitting or predicting. The operator then skips the LLM and runs that code.

## Worked example

This example optimises one `sem_gen_features` operator on the training part of a dataset, then refits with the best code and evaluates **once** on the held-out test set.

```python
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sempipes
from sempipes.optimisers import MonteCarloTreeSearch, optimise_colopro

dataset = skrub.datasets.fetch_midwest_survey()
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.3, random_state=0)

# 1. Build the pipeline with variables WITHOUT values (no LLM calls yet)
responses = sempipes.as_X(skrub.var("responses"), dataset.metadata["description"])
labels = sempipes.as_y(skrub.var("labels"), dataset.metadata["target"])

with_features = responses.sem_gen_features(
    nl_prompt="Compute demographics-related features, use your knowledge about the US and the Midwest.",
    name="demographic_features",
    how_many=5,
)
encoded = with_features.skb.apply(skrub.TableVectorizer())
pipeline = encoded.skb.apply(HistGradientBoostingClassifier(), y=labels)

# 2. Optimise on the TRAINING data only
outcomes = optimise_colopro(
    pipeline,
    num_trials=8,
    search=MonteCarloTreeSearch(),      # a fresh search object for every call
    scoring="accuracy",
    cv=3,
    additional_env_variables={"responses": X_train, "labels": y_train},
    run_name="midwest",
)

# 3. Inspect the best trial (ties: prefer the earlier trial)
best = max(outcomes, key=lambda o: (o.score, -o.search_node.trial))
print(f"best CV accuracy: {best.score:.3f} (trial {best.search_node.trial})")
print(best.states.get("demographic_features")["generated_code"])

# 4. Refit with the best code on all training data, evaluate once on the test set
env_train = pipeline.skb.get_data()
env_train.update({"responses": X_train, "labels": y_train})
env_train["sempipes_prefitted_state__demographic_features"] = best.states.get("demographic_features")

learner = pipeline.skb.make_learner()
learner.fit(env_train)                       # no LLM call: the prefitted code is used

env_test = pipeline.skb.get_data()
env_test["responses"] = X_test
print("test accuracy:", accuracy_score(y_test, learner.predict(env_test)))
```

For a pipeline with several optimisable operators, set one `sempipes_prefitted_state__<name>` entry per operator from the same `best` outcome. The operators' code was scored together, so keep it together.

## Trajectories

At the **end** of a run, `optimise_colopro` writes a JSON file to `.sempipes_trajectories/<run_name or colopro>_<YYYYmmdd_HHMMSS>_<id>.json` in the current working directory. The file contains the SemPipes config, the optimiser arguments and all outcomes, including every generated program. If the run crashes, nothing is written.

```python
from pathlib import Path
from sempipes.optimisers.trajectory import load_trajectory_from_json

path = sorted(Path(".sempipes_trajectories").glob("midwest_*.json"))[-1]
trajectory = load_trajectory_from_json(path)

best = max(trajectory.outcomes, key=lambda o: (o.score, -o.search_node.trial))
print(trajectory.optimizer_args)
print(best.score, best.states.get("demographic_features")["generated_code"])
```

`trajectory.sempipes_config` holds the `Config` used, `trajectory.optimizer_args` the arguments, and `trajectory.outcomes` the `Outcome` objects. You can also save a trajectory yourself with `save_trajectory_as_json(trajectory, run_name=None, output_folder=None)`.

The notebook [`examples/notebooks/demo_view_trajectory.ipynb`](../examples/notebooks/demo_view_trajectory.ipynb) draws the search tree (trial number and score per node, best trial in red) and prints the best generated code of every optimised operator. It loads the most recent file in `.sempipes_trajectories/`, searching the notebook folder and its parents, so run [`examples/12_optimise_colopro.py`](../examples/12_optimise_colopro.py) (or an optimisation notebook) first. It needs no Graphviz installation.

## Cost guidance

A rough budget per run:

* **LLM calls:** trial 0 makes one code-generation conversation per optimisable operator. Every later trial makes one conversation per evolved operator, plus re-generated downstream operators. Each conversation is 1 to `max_retries_for_code_gen` requests, and the prompts grow with the memory. Roughly `(num_trials + number_of_operators) × (1–5)` requests.
* **Model training:** each trial runs `cv` model fits, or `cv × num_hpo_iterations_per_trial` if the pipeline has choices. Each evolution step also runs the pipeline up to the evolved operator once on the full training data.
* **Hidden extra cost:** *non-optimisable* semantic operators in the same pipeline (e.g. `sem_fillna`, `sem_select`, per-row `sem_extract_features`) are refitted in every evolution step and in **every CV fold of every trial**, and they call the LLM each time. Run such steps once beforehand (`.skb.eval()`) and feed their output into the optimised pipeline as data.
* Start small: `num_trials=5`, `cv=3`, a few hundred rows, then scale up.

## Pitfalls

* **Always pass a fresh search object.** The default `search=TreeSearch()` is created once when the module is imported and shared between calls. A second call without `search=` continues with the first run's outcomes. Write `search=TreeSearch()` or `search=MonteCarloTreeSearch()` explicitly every time.
* **Higher must be better.** All search policies and the randomized search maximise the score. For regression use `"neg_mean_squared_error"`, `"neg_root_mean_squared_error"`, `"r2"`, and so on.
* **Scores are CV scores on the training data.** With few rows and many trials you can overfit the CV. Always confirm on a held-out test set (step 4 above).
* **Generated code can crash on unseen data.** The code of every trial has only been run on the training data. If the test data contains a value it has never seen, e.g. a free-text ZIP answer "No home" where the code calls `int(...)`, applying the state fails. Describe such quirks of the data in `nl_prompt`, and fall back to the next-best outcome if the best one fails. [`examples/12_optimise_colopro.py`](../examples/12_optimise_colopro.py) shows both.
* **Parallel workers don't see `update_config`.** Cross-validation runs in separate processes (`n_jobs_for_evaluation=-1`, and always for the randomized search). The [config is a ContextVar](configuration.md#caveat-the-config-is-per-thread--per-context), so the workers use the **default** config. This doesn't matter for the optimisable operators, whose code is frozen during scoring. It does matter for other semantic operators that call the LLM inside the folds. In that case, set `n_jobs_for_evaluation=1` (pipelines with choices can't be made sequential).
* **Operator names** must be unique and must not contain `__`.
* **Mild optimism in the CV score:** code is generated from (samples of) the *full* training data and then cross-validated with that code frozen. The LLM may therefore have seen rows that end up in the validation folds. The effect is usually small, but it is one more reason to confirm on a held-out test set.
* `EvolutionarySearch` assumes positive scores. With negative scorers (`neg_*`) use `TreeSearch` or `MonteCarloTreeSearch`.

## See also

* Example script: [`examples/12_optimise_colopro.py`](../examples/12_optimise_colopro.py)
* Notebooks: [`examples/notebooks/demo__optimise_semantic_operator.ipynb`](../examples/notebooks/demo__optimise_semantic_operator.ipynb), [`examples/notebooks/demo__optimise_multiple_semantic_operators.ipynb`](../examples/notebooks/demo__optimise_multiple_semantic_operators.ipynb), [`examples/notebooks/demo_view_trajectory.ipynb`](../examples/notebooks/demo_view_trajectory.ipynb)
* Operators: [`sem_gen_features`](operators/sem_gen_features.md), [`sem_agg_features`](operators/sem_agg_features.md), [`sem_extract_features`](operators/sem_extract_features.md), [`sem_augment`](operators/sem_augment.md)
