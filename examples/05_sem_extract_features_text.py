"""sem_extract_features (text): turn free text into new categorical features with an LLM.

What the operator does
    ``DataOp.sem_extract_features(nl_prompt, input_columns, name, output_columns=...)`` adds new columns derived from
    text, image or audio columns. ``output_columns`` maps each new column name to a short instruction. If you leave
    it out, the LLM proposes the features itself. The operator has two processing modes:
    * ``generate_via_code=False``: the LLM reads every row and answers directly (one request per row).
    * ``generate_via_code=True`` (the default): the LLM writes Python code, typically using a pretrained
      ``transformers`` model, which then runs locally on all rows. That is cheaper for large data, but it downloads
      models and runs them on your machine.

What this example shows
    100 tweets from skrub's ``toxicity`` dataset. The LLM labels each tweet with three intermediate features (insult
    level, target of the attack, sentiment), deliberately *not* the toxicity label itself. We then compare the
    cross-validated accuracy of a classifier on the raw text alone with one on raw text plus LLM features.

How to run (from the repository root)
    poetry run python examples/05_sem_extract_features_text.py

Runtime / cost
    About 1-2 minutes, with 100 LLM calls (one per tweet, sent in batches of 20).
"""

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

import sempipes  # importing sempipes adds the .sem_* methods to skrub DataOps

N_TWEETS = 100
OUTPUT_COLUMNS = {
    "insult_level": "Classify the insulting or profane language as exactly one of: none, mild, strong.",
    "attack_target": "Who does the tweet attack? Answer exactly one of: nobody, a person, a group.",
    "sentiment": "Classify the overall sentiment as exactly one of: positive, neutral, negative.",
}


def main() -> None:
    dataset = skrub.datasets.fetch_toxicity()
    tweets_df = dataset.X.sample(n=N_TWEETS, random_state=0)
    is_toxic = (dataset.y.loc[tweets_df.index] == "Toxic").astype(int)

    tweets = skrub.var("tweets")
    with_features = tweets.sem_extract_features(
        nl_prompt="Extract features that help to predict whether a tweet is toxic. Do not output the toxicity itself.",
        input_columns=["text"],
        name="tweet_features",
        output_columns=OUTPUT_COLUMNS,
        generate_via_code=False,  # one LLM call per row; True would make the LLM write local extraction code
    )

    # Operators such as sem_extract_features also create a few hidden helper variables that the optimiser uses
    # (see 12_optimise_colopro.py). get_data() returns their default values, and we add our own data to them.
    env = with_features.skb.get_data()
    env["tweets"] = tweets_df.copy()
    features = with_features.skb.eval(env)

    print("=== Tweets with LLM-extracted features ===")
    print(features.head(8).to_string(max_colwidth=60))
    print("\n=== How the extracted insult level relates to the true label (0 = not toxic, 1 = toxic) ===")
    print(pd.crosstab(features["insult_level"], is_toxic))

    # The extraction looks at one row at a time and never at the labels. Running it once on all tweets before
    # cross-validation therefore leaks no information, and it saves LLM calls (no re-extraction per fold).
    model = make_pipeline(skrub.TableVectorizer(), HistGradientBoostingClassifier(random_state=0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print("\n=== 5-fold cross-validated accuracy ===")
    for title, columns in [("text only", ["text"]), ("text + LLM features", ["text", *OUTPUT_COLUMNS])]:
        scores = cross_val_score(model, features[columns], is_toxic, cv=cv, scoring="accuracy")
        print(f"{title:<20} {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()
