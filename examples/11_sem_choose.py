"""sem_choose: let an LLM propose the hyperparameter search space.

What the operator does
    ``sempipes.sem_choose(name, param=description, ...)`` describes in natural language which values of an estimator's
    hyperparameters are worth trying. ``DataOp.skb.apply_with_sem_choose(estimator, choices=..., y=...)`` works like
    skrub's ``.skb.apply`` but first asks the LLM to turn each description into a ``skrub.choose_from([...])`` list.
    This happens *immediately*, while the pipeline is defined (one LLM call per parameter). The result is an
    ordinary skrub search space, so any skrub search (grid, randomized, or the SemPipes optimiser) can tune it.

What this example shows
    We predict the US census region of respondents of skrub's Midwest survey with a gradient-boosting classifier.
    The LLM proposes learning rates and tree sizes. We print the proposed search space, run a small randomized
    search (6 candidates, 3-fold CV) on a training split and evaluate the best model on a held-out test split.

How to run (from the repository root)
    poetry run python examples/11_sem_choose.py

Runtime / cost
    About 1-2 minutes, with 2 LLM calls (one per parameter). The search itself runs locally.
"""

import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

import sempipes  # importing sempipes adds .skb.apply_with_sem_choose to skrub DataOps


def main() -> None:
    dataset = skrub.datasets.fetch_midwest_survey()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.25, stratify=dataset.y, random_state=0
    )

    responses = skrub.var("responses").skb.mark_as_X()
    regions = skrub.var("regions").skb.mark_as_y()
    encoded = responses.skb.apply(skrub.TableVectorizer())

    # The LLM is called right here, while we define the pipeline.
    model = encoded.skb.apply_with_sem_choose(
        HistGradientBoostingClassifier(random_state=0),
        y=regions,
        choices=sempipes.sem_choose(
            name="hgb",
            learning_rate="A few learning rates that work well for a small tabular dataset with ~2000 rows.",
            max_leaf_nodes="Tree sizes from small to large, to trade off underfitting and overfitting.",
        ),
    )

    print("\n=== Search space proposed by the LLM ===")
    print(model.skb.describe_param_grid())

    # get_data() returns default values for SemPipes' hidden helper variables. We add our data to them.
    env = model.skb.get_data()
    search = model.skb.make_randomized_search(n_iter=6, cv=3, scoring="accuracy", random_state=0)
    search.fit({**env, "responses": X_train, "regions": y_train})

    print("=== Randomized search results (3-fold CV accuracy on the training split) ===")
    print(search.results_.to_string())
    print(f"\nBest parameters:  {search.best_learner_.describe_params()}")
    print(f"Best CV accuracy: {search.best_score_:.3f}")
    # The search refits the best candidate on the whole training split. We score it on unseen data.
    print(f"Test accuracy:    {search.score({**env, 'responses': X_test, 'regions': y_test}):.3f}")


if __name__ == "__main__":
    main()
