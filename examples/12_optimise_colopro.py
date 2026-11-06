"""optimise_colopro: automatically improve the code that semantic operators generate.

What the optimiser does
    ``sempipes.optimisers.optimise_colopro(pipeline, num_trials, search=..., scoring=..., cv=...)`` treats the code
    that a code-generating operator (here ``sem_gen_features``) produced as a *state* that can be improved. In every
    trial it asks the LLM for a new version of the operator's code and shows it earlier versions with their
    cross-validation scores as feedback ("optimisation by prompting"). A search policy, here Monte Carlo tree
    search, decides which earlier version to build on. Hyperparameters defined with ``sem_choose`` are tuned with a
    small random search in every trial. The result is a list of outcomes, each with a ``score`` and ``states``.

What this example shows
    We predict the US census region of Midwest-survey respondents. We optimise on a training split only. Then we
    hand the best generated code to the pipeline through the variable ``sempipes_prefitted_state__<operator name>``
    and evaluate it on a held-out test split. This step makes no LLM calls, because the operator reuses the state
    instead of generating new code. For comparison we evaluate the same pipeline without generated features.

How to run (from the repository root)
    poetry run python examples/12_optimise_colopro.py

Runtime / cost
    About 5-10 minutes, with roughly 8-15 LLM calls (1 for sem_choose, then 1 code generation per trial plus retries).
    The search trajectory is saved as JSON in ``.sempipes_trajectories/`` in the current directory.
"""

import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps
from sempipes.optimisers import MonteCarloTreeSearch, optimise_colopro

OPERATOR = "demographic_features"
NO_FEATURES = {"generated_code": "def _sem_gen_features(df):\n    return df\n"}  # state that adds no features


def build_pipeline(description: str, target_description: str) -> skrub.DataOp:
    # The descriptions are part of the context the optimiser gives the LLM.
    responses = sempipes.as_X(skrub.var("responses"), description)
    regions = sempipes.as_y(skrub.var("regions"), target_description)
    with_features = responses.sem_gen_features(
        nl_prompt="Compute features that help to predict the US census region of a respondent from their answers "
        "and demographics. Use your knowledge about US geography and pay special attention to the ZIP code. "
        # Without this hint, the generated code tends to call int() on the ZIP answers, which works on the training
        # split but crashes on a test answer such as 'No home'. Telling the LLM about such quirks makes the code robust.
        "The ZIP code answers are free text: they can contain ZIP+4 codes, several codes or non-ZIP answers such as "
        "'No home', so parse them defensively and never fail on unexpected values.",
        name=OPERATOR,
        how_many=5,
    )
    encoded = with_features.skb.apply(skrub.TableVectorizer())
    return encoded.skb.apply_with_sem_choose(  # one LLM call, right now
        HistGradientBoostingClassifier(random_state=0),
        y=regions,
        choices=sempipes.sem_choose(name="hgb", learning_rate="A few learning rates that work well on small data."),
    )


def main() -> None:
    dataset = skrub.datasets.fetch_midwest_survey()
    X, y = dataset.X.iloc[:1200], dataset.y.iloc[:1200]  # a subset keeps every trial fast
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    pipeline = build_pipeline(dataset.metadata["description"], dataset.metadata["target"])

    outcomes = optimise_colopro(
        pipeline,
        num_trials=6,
        search=MonteCarloTreeSearch(),  # always pass a fresh search object: it stores the search tree of one run
        scoring="accuracy",
        cv=3,
        num_hpo_iterations_per_trial=1,
        additional_env_variables={"responses": X_train, "regions": y_train},  # the optimiser only sees training data
        run_name="midwest_example",
    )
    print("\n=== Cross-validation accuracy per trial ===")
    for outcome in outcomes:
        print(f"trial {outcome.search_node.trial}: {outcome.score:.3f}")

    def test_accuracy(state: dict) -> float:
        env = pipeline.skb.get_data()
        env[f"sempipes_prefitted_state__{OPERATOR}"] = state  # reuse this code instead of asking the LLM
        learner = pipeline.skb.make_learner()  # uses the first (default) value of every sem_choose parameter
        learner.fit({**env, "responses": X_train, "regions": y_train})
        return accuracy_score(y_test, learner.predict({**env, "responses": X_test}))

    print("\n=== Held-out test accuracy ===")
    print(f"{'Without generated features:':<34}{test_accuracy(NO_FEATURES):.3f}")
    # Generated code was only ever run on training data. On unseen test data it can still fail, e.g. when it
    # converts a column to int and the test data contains a value it has never seen. Then we take the next-best trial.
    for outcome in sorted(outcomes, key=lambda outcome: outcome.score, reverse=True):
        state = outcome.states.get(OPERATOR)  # a dict holding the generated code
        try:
            accuracy = test_accuracy(state)
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(
                f"The code of trial {outcome.search_node.trial} fails on the test data ({error}), trying the next one"
            )
            continue
        print(f"{f'With best features (trial {outcome.search_node.trial}):':<34}{accuracy:.3f}")
        print("\n=== Best generated code ===")
        print(state["generated_code"])
        break
    else:
        print("The generated code of every trial fails on the test data. Describe the quirks of the data in nl_prompt")
        print("(see the ZIP code hint above) or optimise again.")


if __name__ == "__main__":
    main()
