"""sem_select: let an LLM decide which columns a model may use.

What the operator does
    ``DataOp.sem_select(nl_prompt)`` looks at every column on its own (name, dtype, ten sample values and a few
    statistics) and asks the LLM whether to keep it, following your natural-language instructions. The decision
    is made once, when the pipeline is fitted. Afterwards the same columns are selected for any new data.

What this example shows
    We predict the annual salary of public employees with skrub's ``employee_salaries`` dataset. The LLM should
    keep the predictive columns and drop protected attributes such as gender. We print the kept columns and
    compare the test R² of a regressor trained on all columns with one trained on the LLM-selected columns.

How to run (from the repository root)
    poetry run python examples/01_sem_select.py

Runtime / cost
    About 1 minute, with 8 small LLM calls (one per input column).
"""

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

SELECTION_PROMPT = (
    "Keep columns that help to predict the annual salary of an employee. "
    "Drop protected attributes that must not influence salary decisions, such as gender."
)


def build_pipeline(use_sem_select: bool) -> skrub.DataOp:
    # The variables have no value yet. We pass the data later, when we fit and evaluate the learner.
    employees = skrub.var("employees").skb.mark_as_X()
    salaries = skrub.var("salaries").skb.mark_as_y()

    features = employees
    if use_sem_select:
        # The LLM-based selection is an ordinary pipeline step. We name it so we can inspect it later.
        features = employees.sem_select(SELECTION_PROMPT).skb.set_name("selected_columns")

    encoded = features.skb.apply(skrub.TableVectorizer())
    return encoded.skb.apply(HistGradientBoostingRegressor(random_state=0), y=salaries)


def main() -> None:
    dataset = skrub.datasets.fetch_employee_salaries()
    data = {"employees": dataset.X, "salaries": dataset.y}
    print(f"Input columns: {list(dataset.X.columns)}")

    for use_sem_select in (False, True):
        pipeline = build_pipeline(use_sem_select)
        # skrub splits the data into train and test environments. random_state=0 makes both runs use the same split.
        split = pipeline.skb.train_test_split(data, test_size=0.25, random_state=0)
        learner = pipeline.skb.make_learner()
        learner.fit(split["train"])  # sem_select queries the LLM here, once per column

        print("\n=== " + ("With sem_select" if use_sem_select else "Baseline: all columns") + " ===")
        if use_sem_select:
            # Run only the part of the fitted pipeline up to the selection step and look at the resulting columns.
            selected = learner.truncated_after("selected_columns").transform(split["test"])
            print(f"Kept columns:    {list(selected.columns)}")
            print(f"Dropped columns: {sorted(set(dataset.X.columns) - set(selected.columns))}")
        print(f"Test R²: {learner.score(split['test']):.3f}")


if __name__ == "__main__":
    main()
