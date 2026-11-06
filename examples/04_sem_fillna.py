"""sem_fillna: impute missing values, either directly with an LLM or with an LLM-chosen model.

What the operator does
    ``DataOp.sem_fillna(target_column, nl_prompt, impute_with_existing_values_only, with_llm_only=False)`` fills the
    missing values of one column. It has two modes:
    * ``with_llm_only=True``: one LLM request per missing value. The LLM sees the other columns of the row and the
      known values of the column and answers with the value to fill in. This suits small amounts of data where
      world knowledge helps, for example knowing which city an address is in.
    * model-based (the default): the LLM writes code for five candidate imputation models. SemPipes scores them on
      the known values and fits the best one, which then fills the gaps. Few LLM calls, and it scales to large tables.
      This mode requires ``impute_with_existing_values_only=True``.

What this example shows
    Part A: restaurant records (name, address, phone, cuisine). We hide the city of 30 test restaurants, let the LLM
    fill them in and measure accuracy against the true values.
    Part B: 600 employee records. We hide the department of 60 employees and impute it with the model-based mode.

How to run (from the repository root)
    poetry run python examples/04_sem_fillna.py

Runtime / cost
    About 2-3 minutes. Part A uses 30 LLM calls (one per missing value). Part B uses 1 code-generation call (plus
    retries) and a few seconds of model training.
"""

from pathlib import Path

import pandas as pd
import skrub

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

DATA_DIR = Path(__file__).parent / "data"
SEED = 0


def hide(df: pd.DataFrame, column: str, rows: pd.Index) -> pd.DataFrame:
    """Return a copy of `df` in which `column` is missing for the given rows."""
    masked = df.copy()
    masked.loc[rows, column] = None
    return masked


def part_a_llm_only() -> None:
    print("=== Part A: LLM-only imputation of restaurant cities ===")
    train = pd.read_csv(DATA_DIR / "restaurants" / "train.csv")
    test = pd.read_csv(DATA_DIR / "restaurants" / "test.csv")
    restaurants = pd.concat([train, test], ignore_index=True)
    # We only hide cities of test rows. The training rows show the LLM which city names exist.
    hidden = restaurants.iloc[len(train) :].sample(n=30, random_state=SEED).index

    filled = skrub.var("restaurants").sem_fillna(
        target_column="city",
        nl_prompt="Infer the city of the restaurant from its street address, phone area code and name.",
        impute_with_existing_values_only=True,  # only answer with city names that already occur in the column
        with_llm_only=True,
    )
    result = filled.skb.eval({"restaurants": hide(restaurants, "city", hidden)})

    truth, imputed = restaurants.loc[hidden, "city"], result.loc[hidden, "city"].str.strip().str.lower()
    print(pd.DataFrame({"address": restaurants.loc[hidden, "addr"], "true city": truth, "imputed": imputed}).head(10))
    # Note: the data uses both 'new york' and 'new york city', so some 'errors' are really just different spellings.
    print(f"Accuracy on {len(hidden)} hidden cities: {(truth == imputed).mean():.0%}\n")


def part_b_model_based() -> None:
    print("=== Part B: model-based imputation of employee departments ===")
    columns = ["department", "division", "employee_position_title", "assignment_category", "year_first_hired"]
    employees = skrub.datasets.fetch_employee_salaries().X[columns].sample(n=600, random_state=SEED)
    employees = employees.reset_index(drop=True)
    hidden = employees.sample(n=60, random_state=SEED).index

    filled = skrub.var("employees").sem_fillna(
        target_column="department",
        # The LLM writes the candidate models. Asking for fast scikit-learn classifiers keeps this example quick.
        # Without this hint the LLM may propose AutoGluon, which trains for several minutes.
        nl_prompt="Predict the department from the division and job title. Use a very fast method: only "
        "light-weight scikit-learn classifiers such as random forests or logistic regression, no AutoGluon.",
        impute_with_existing_values_only=True,
    )
    result = filled.skb.eval({"employees": hide(employees, "department", hidden)})

    truth, imputed = employees.loc[hidden, "department"], result.loc[hidden, "department"]
    print(pd.DataFrame({"division": employees.loc[hidden, "division"], "true": truth, "imputed": imputed}).head(10))
    majority = (truth == employees.drop(index=hidden)["department"].mode()[0]).mean()
    print(f"Accuracy on {len(hidden)} hidden departments: {(truth == imputed).mean():.0%}")
    print(f"(Baseline that always predicts the most frequent department: {majority:.0%})")


def main() -> None:
    part_a_llm_only()
    part_b_model_based()


if __name__ == "__main__":
    main()
