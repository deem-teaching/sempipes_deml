"""sem_refine: unify inconsistent categorical values (typos, synonyms, languages).

What the operator does
    ``DataOp.sem_refine(target_column, nl_prompt, refine_with_existing_values_only)`` sends the value counts of one
    column to the LLM and asks it to write a ``refine_column(df)`` function. That function is usually a mapping
    from every messy variant to a canonical value. It deduplicates near-duplicates, fixes typos, translates
    values and marks outliers. With ``refine_with_existing_values_only=True`` the LLM may only map to values that
    already occur in the column. With ``False`` it may introduce new, cleaner values.

What this example shows
    An inline table of addresses with messy city names (typos, local spellings, country suffixes) and messy
    country codes. We refine both columns and print the value counts before and after, plus a before/after table.

How to run (from the repository root)
    poetry run python examples/03_sem_refine.py

Runtime / cost
    Under a minute, with 2 LLM calls (one per refined column, plus retries if the generated code fails).
"""

import pandas as pd
import skrub

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

# fmt: off
ADDRESSES = pd.DataFrame(
    {
        "city": [
            "Berlin", "berlin", "Berlin, Germany", "Berln", "Munich", "München", "Muenchen", "munich ",
            "Rome", "Roma", "Rome, Italy", "Romq", "Paris", "paris", "Paris (FR)", "Pariss",
        ],
        "country_code": [
            "DE", "de", "GER", "DEU", " DE", "Germany", "DE", "de ",
            "IT", "ITA", "it", "Italy", "FR", "FRA", "fr", "France",
        ],
    }
)
# fmt: on


def main() -> None:
    addresses = skrub.var("addresses")

    refined = addresses.sem_refine(
        target_column="city",
        nl_prompt="Fix typos, remove country names and use the English name of each city.",
        refine_with_existing_values_only=False,  # allow new canonical values, e.g. 'Munich' for 'München'
    )
    refined = refined.sem_refine(
        target_column="country_code",
        nl_prompt="Map every value to its ISO 3166-1 alpha-2 country code in upper case, e.g. 'DE'.",
        refine_with_existing_values_only=False,
    )

    # eval() fits the pipeline on this data (LLM calls) and returns the refined table.
    result = refined.skb.eval({"addresses": ADDRESSES})

    for column in ["city", "country_code"]:
        print(f"\n=== {column}: {ADDRESSES[column].nunique()} distinct values before refinement ===")
        print(ADDRESSES[column].value_counts().to_string())
        print(f"\n=== {column}: {result[column].nunique()} distinct values after refinement ===")
        print(result[column].value_counts().to_string())

    print("\n=== Row by row ===")
    print(pd.concat([ADDRESSES, result], axis=1, keys=["before", "after"]).to_string())


if __name__ == "__main__":
    main()
