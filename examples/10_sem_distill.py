"""sem_distill: shrink a training set to a small, representative subset with LLM-written code.

What the operator does
    ``DataOp.sem_distill(nl_prompt, number_of_rows)`` asks the LLM to write a ``distill_data(df, number_of_rows)``
    function that returns exactly ``number_of_rows`` rows that represent the full data well. Typical techniques are
    stratified sampling, clustering-based coreset selection or synthetic prototypes. Like ``sem_augment``, it only
    acts while the pipeline is *fitted*. At prediction time the data passes through unchanged.

What this example shows
    We train a salary regressor on skrub's ``employee_salaries`` data three times: on all 8000 training rows, on a
    random 10 % sample and on an LLM-distilled 10 % subset. All three are evaluated on the official held-out test
    split. A good distillation should beat the random sample of the same size.

How to run (from the repository root)
    poetry run python examples/10_sem_distill.py

Runtime / cost
    About 1-2 minutes, with 1 LLM call (plus retries if the generated code fails validation).
"""

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

N_DISTILLED = 800


def sample_when_fitting(df, mode: str):
    """Baseline: a random sample of the same size, taken (like sem_distill) only while fitting."""
    return df.sample(n=N_DISTILLED, random_state=0) if "fit" in mode else df


def build_pipeline(training_rows: str) -> skrub.DataOp:
    salaries = skrub.var("salaries")  # the full table, target column included

    if training_rows == "distilled":
        salaries = salaries.sem_distill(
            nl_prompt="Select a representative subset for predicting 'current_annual_salary': preserve the salary "
            "distribution and cover all departments and position titles as well as possible. Use fast methods "
            "only (e.g. stratified sampling or k-means on numeric features), do not train neural networks.",
            number_of_rows=N_DISTILLED,
        )
    elif training_rows == "random":
        # skrub.deferred runs a plain function inside the pipeline. eval_mode() says whether we fit or predict.
        salaries = skrub.deferred(sample_when_fitting)(salaries, skrub.eval_mode())

    # errors="ignore": at prediction time the input table has no salary column to drop.
    X = salaries.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
    y = salaries["current_annual_salary"].skb.mark_as_y()
    return X.skb.apply(skrub.TableVectorizer()).skb.apply(HistGradientBoostingRegressor(random_state=0), y=y)


def main() -> None:
    train = skrub.datasets.fetch_employee_salaries(split="train").employee_salaries
    test = skrub.datasets.fetch_employee_salaries(split="test")

    for training_rows, size in [("full", len(train)), ("random", N_DISTILLED), ("distilled", N_DISTILLED)]:
        pipeline = build_pipeline(training_rows)
        learner = pipeline.skb.make_learner()
        learner.fit({"salaries": train})  # sem_distill generates and runs its code here
        predictions = learner.predict({"salaries": test.X})
        print(f"=== Training on {training_rows} data ({size} rows) ===")
        print(f"Test R²: {r2_score(test.y, predictions):.3f}\n")


if __name__ == "__main__":
    main()
