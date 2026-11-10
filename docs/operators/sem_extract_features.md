# `sem_extract_features`: derive features from text, images and audio

`sem_extract_features` turns unstructured columns (free text, image paths or URLs, audio files) into new tabular features. You describe what you want, either as a single prompt or as an explicit `{feature name: description}` dictionary. There are two modes:

* **Code mode (default, `generate_via_code=True`):** the LLM writes Python code that uses pretrained HuggingFace `transformers` models (zero-shot) or plain Python/regex to compute the features. The code runs **locally**, in a separate process, on training and test data. The LLM is not called again after fitting. This mode is **optimisable**.
* **Per-row mode (`generate_via_code=False`):** every row (text plus images or audio) is sent to a (multimodal) LLM, which answers with the feature values as JSON.

| | Code mode (default) | Per-row mode |
|---|---|---|
| LLM called when fitting | +1 if `output_columns=None` (feature proposal), then 1 code-generation conversation | +1 if `output_columns=None`; then 1 request per training row |
| LLM called when transforming | never; the stored code (and local models) run again | 1 request per row |
| Model used | `llm_for_code_generation` | `llm_for_batch_processing` (feature proposal: `llm_for_code_generation`) |
| Optimisable | **yes** | no |
| Source | `sem_extract_features/with_code.py` | `sem_extract_features/with_llm.py` |

All source files are in `sempipes/operators/sem_extract_features/`. `__init__.py` dispatches between the modes, and `_shared.py` contains modality detection and feature proposal.

## Signature

```python
DataOp.sem_extract_features(
    nl_prompt: str,
    input_columns: list[str],
    name: str,
    output_columns: dict[str, str] | None = None,
    **kwargs,   # generate_via_code: bool = True, print_code_to_console: bool = False
) -> DataOp
```

## Arguments

| Name | Type | Default | Description |
|---|---|---|---|
| `nl_prompt` | `str` | required | Overall instructions. In code mode they also constrain the code (e.g. "do not use the transformers library, use regular expressions"). In per-row mode with explicit `output_columns`, this prompt is **not** sent; only the per-feature descriptions are. |
| `input_columns` | `list[str]` | required | Columns to extract from (text, image paths/URLs, audio paths/URLs). |
| `name` | `str` | required | Unique name of this operator in the pipeline, used by the [optimiser](../optimizer.md). Must not contain `__`. |
| `output_columns` | `dict[str, str] \| None` | `None` | `{new column name: what to compute}`. With `None`, the LLM first **proposes** features (see below). |
| `generate_via_code` | `bool` (keyword) | `True` | `True` selects code mode; `False` selects per-row mode. |
| `print_code_to_console` | `bool` (keyword) | `False` | Code mode only. The generated code is printed before every validation run either way; this flag only changes the header above it. |

The result is the **original table plus the new columns**. The input columns are kept, so drop them (`.drop(columns=[...])`) before a model that can't handle them.

## Modalities

Each input column gets a modality, detected from its values:

| Modality | Rule (applied to a random 20% sample of the column) |
|---|---|
| image | **every** sampled value is a string ending in `.jpg`, `.jpeg`, `.png`, `.bmp` or `.gif` |
| audio | **every** sampled value is a string ending in `.wav`, `.mp3`, `.flac`, `.aac` or `.ogg` |
| text | everything else |

Values can be **local file paths** or **`http(s)://` URLs**. In per-row mode and during feature proposal, local files are read and sent as base64 data URLs, and URLs are passed to the provider as they are. In code mode, the generated code opens the files itself (e.g. with `PIL`).

Watch out:

* The check is **case-sensitive**: `photo.JPG` counts as text.
* URLs with query strings (`...jpg?w=200`) don't end in the extension and count as text.
* A single missing value in the sample makes the whole column count as text.
* Relative paths are resolved against the **current working directory**.
* Images and audio in per-row mode need a **multimodal** batch model. Feature proposal also sends two sample images or audio files to the code-generation model. The default `gemini/gemini-2.5-flash` supports both.

## How it works

### Feature proposal (`output_columns=None`, both modes)

The code-generation model receives your `nl_prompt`, the input column names and **two non-missing sample values per column** (images and audio attached). It answers with a JSON list of `{"feature_name", "feature_prompt", "input_columns"}` objects. Entries with any other keys are dropped and logged ("Unable to parse some of the suggested features."). The proposal is made once per fit, so different fits (e.g. different CV folds) can produce different feature sets.

### Code mode

1. The code-generation model receives, per input column: dtype, modality, **10 sample values** (`random_state=42`) and, for text, the maximum and mean number of words. It also gets the list of features to compute and your `nl_prompt`. It is asked for `extract_features(df) -> df` that uses zero-shot `transformers` models (unless you forbid them), runs in batches with a progress bar, and prefers MPS, then CUDA, then CPU.
2. The code is validated **in a separate Python process** on the **first 50 rows**. It must return all original columns plus all requested new columns, the same number of rows and unique column names. Errors go back to the LLM, up to `max_retries_for_code_gen` attempts.
3. At transform time, the stored code runs again in a subprocess on the full input. Models are downloaded from the HuggingFace Hub on first use (set `HF_TOKEN` for gated models, see [installation](../installation.md#4-api-keys)).
4. **If every attempt fails**, SemPipes logs an error and falls back to an identity function. The step then **adds no columns**, and a later `df["my_feature"]` fails with a `KeyError`.

### Per-row mode

At fit time, SemPipes detects the modalities and, if needed, proposes features. At transform time, it sends one request per row containing the row's input values (text inline, images and audio attached) and asks for a JSON object with one key per feature. Unparsable answers are retried up to 5 rounds. Rows that still fail get the value `"Unknown"` for every feature. If the LLM returns **extra keys**, they become **extra columns**.

## Examples

Per-row mode on a tiny inline table (5 LLM requests, nothing to download):

```python
import pandas as pd
import skrub

import sempipes

reviews = pd.DataFrame(
    {
        "review": [
            "Great battery life, but the screen scratches easily.",
            "Stopped working after two days. Avoid!",
            "Does what it says. Delivery was fast.",
            "Absolutely love it, best purchase this year.",
            "The manual is only in Chinese and the app keeps crashing.",
        ]
    }
)
data = skrub.var("reviews", reviews)

with_features = data.sem_extract_features(
    nl_prompt="Extract features that describe customer satisfaction.",
    input_columns=["review"],
    name="review_features",
    output_columns={
        "sentiment": "Overall sentiment of the review: 'positive', 'negative' or 'mixed'.",
        "mentions_defect": "1 if the review mentions a broken or defective product, else 0.",
    },
    generate_via_code=False,
)
print(with_features.skb.preview())
```

Code mode (default) on text. The LLM writes `transformers` code once, and the code runs locally on all rows:

```python
import skrub

import sempipes

dataset = skrub.datasets.fetch_toxicity()
tweets = skrub.var("tweets", dataset.X.head(50))      # code mode needs at least 10 rows

with_features = tweets.sem_extract_features(
    nl_prompt="Extract features that help to detect toxic language.",
    input_columns=["text"],
    name="tweet_features",
    output_columns={"maybe_toxic": "1 if the tweet is toxic, else 0."},
)
print(with_features.skb.preview()["maybe_toxic"].value_counts())
```

Images (per-row; run from the repository root):

```python
import pandas as pd
import skrub

import sempipes

styles = pd.read_csv("examples/data/fashion/styles.csv")
styles["image_path"] = "examples/data/fashion/" + styles["image_path"]   # e.g. images/15970.jpg

products = skrub.var("products", styles[["id", "image_path"]])
described = products.sem_extract_features(
    nl_prompt="Describe the product in the photo.",
    input_columns=["image_path"],
    name="image_features",
    output_columns={"dominant_colour": "The dominant colour of the product, one word."},
    generate_via_code=False,
)
print(described.skb.preview())
```

## See also

* Example scripts: [`examples/05_sem_extract_features_text.py`](../../examples/05_sem_extract_features_text.py), [`examples/06_sem_extract_features_images.py`](../../examples/06_sem_extract_features_images.py)
* Notebook: [`examples/notebooks/demo_sem_extract_features.ipynb`](../../examples/notebooks/demo_sem_extract_features.ipynb)
* Related: [`sem_gen_features`](sem_gen_features.md) (features from *tabular* columns), the [optimiser](../optimizer.md) (code mode can be optimised), [configuration](../configuration.md) (choosing multimodal models).
