# Installation

This page gets you from a fresh machine to a working `import sempipes`. Expect the first install to take a while: SemPipes pulls in PyTorch, HuggingFace `transformers`, AutoGluon and SDV, which together need **several GB** of disk space.

## 1. Python 3.11 or 3.12

SemPipes requires **Python 3.11 or 3.12** (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`). Python 3.13 is **not** supported, and many systems now ship it as their default, so check first:

```bash
python3 --version
python3.12 --version   # is a 3.12 interpreter available?
```

If you don't have 3.11 or 3.12, install one of them alongside your system Python:

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install python@3.12` |
| Ubuntu / Debian | `sudo apt install python3.12 python3.12-venv` (older Ubuntu releases need the *deadsnakes* PPA) |
| Windows | Download the 3.12 installer from [python.org](https://www.python.org/downloads/) |
| conda / mamba | `conda create -n py312 python=3.12 -y`, then `conda activate py312` and run `which python` to find the path |
| pyenv | `pyenv install 3.12`, then `pyenv which python` (inside a directory where 3.12 is selected) or look under `$(pyenv root)/versions/` |

You will hand the interpreter to Poetry in step 3. Either its name (`python3.12`) or its full path works.

## 2. Install Poetry 2

SemPipes uses [Poetry](https://python-poetry.org/) to manage dependencies. The project uses the `[project]` table in `pyproject.toml`, so you need **Poetry 2.x**. Poetry 1.x cannot install it.

```bash
pipx install poetry          # recommended; or: curl -sSL https://install.python-poetry.org | python3 -
poetry --version             # should print 2.x
```

If you already have Poetry 1.x, upgrade it with `pipx upgrade poetry` or `poetry self update`.

## 3. Create the environment and install

From the repository root (the folder that contains `pyproject.toml`):

```bash
poetry env use python3.12    # or the full path to a 3.11/3.12 interpreter
poetry install               # installs SemPipes + all dependencies (several GB, can take 10+ minutes)
```

`poetry install` also installs the development group, which includes `pytest` and `jupyter`.

## 4. API keys

By default SemPipes calls **Gemini 2.5 Flash** through [litellm](https://docs.litellm.ai/). Put your key into a `.env` file in the repository root:

```bash
cp .env.example .env
```

Then edit `.env` and fill in at least one key:

```bash
GEMINI_API_KEY=...   # needed for the default model (gemini/gemini-2.5-flash)
OPENAI_API_KEY=...   # only if you switch to an openai/... model
HF_TOKEN=...         # optional, see below
```

* **Gemini (default):** you can create a free API key in [Google AI Studio](https://aistudio.google.com/apikey). The free tier has rate limits, which matter for operators that send one request per row (see [Troubleshooting](#troubleshooting)).
* **OpenAI:** you need a paid key from the OpenAI platform. You also have to tell SemPipes to use an `openai/...` model (see [configuration.md](configuration.md#provider-examples)).
* **SAIA (GWDG Academic Cloud):** free for members of German universities. Put the key and the base URL into `.env` as `SAIA_API_KEY` and `SAIA_BASE_URL` and switch with `sempipes._set_config(sempipes.get_default_saia_config())`, or make SAIA the default for everything (including the tests) with `SEMPIPES_LLM_PROVIDER=saia` in `.env` (see [configuration.md](configuration.md#using-saia-gwdg-academic-cloud)).
* **`HF_TOKEN` (optional):** `sem_extract_features` in its default code mode runs HuggingFace models locally and downloads them from the HuggingFace Hub on first use. A token is only needed for gated models, and it also raises your download rate limits.
* **Local models via [Ollama](https://ollama.com/):** these need no key. Install Ollama, pull a model (e.g. `ollama pull gpt-oss:20b`) and point SemPipes at it with `api_base` (see [configuration.md](configuration.md#provider-examples)). Small local models often struggle to write working code, so expect more retries.

`import sempipes` calls `load_dotenv()`, which finds the `.env` file in the repository root automatically as long as you work inside the repository folder (scripts, notebooks in `examples/notebooks/`, `poetry run python -c ...` from the root). Variables that are already set in your shell take precedence over `.env`. If you edit `.env` while a Jupyter kernel is running, restart the kernel, because the file is read only at import time.

> **Your data goes to the LLM provider.** Operators send column names, sample values, statistics and, for per-row operators, whole rows to the configured provider. Only use data you are allowed to share with that provider.

## 5. Verify the installation

```bash
poetry run python -c "import sempipes; print(sempipes.get_config())"
```

The first import is **slow**, often tens of seconds on a cold start. `import sempipes` loads torch, transformers, AutoGluon and SDV because generated code may use them. Later imports are faster.

To check that your API key works (this makes one small LLM call):

```bash
poetry run python -c "import sempipes, litellm; r = litellm.completion(model=sempipes.get_config().llm_for_code_generation.name, messages=[{'role': 'user', 'content': 'Say hi'}]); print(r.choices[0].message.content)"
```

## 6. Running code

Run scripts through Poetry from the repository root:

```bash
poetry run python examples/01_sem_select.py
```

Alternatively, activate the environment in your shell. Poetry 2 removed `poetry shell`, so use:

```bash
eval $(poetry env activate)          # bash / zsh
# Invoke-Expression (poetry env activate)   # PowerShell
python examples/01_sem_select.py
```

To use Jupyter:

```bash
poetry run jupyter lab
```

Notebooks live in [`examples/notebooks/`](../examples/notebooks/). Run commands from the repository root: the examples use relative data paths, and SemPipes writes some files into the current working directory (see [concepts.md](concepts.md#files-sempipes-writes)).

### Hardware notes

* A CPU is enough for every operator.
* `sem_extract_features` in code mode runs HuggingFace models locally. The generated code is told to prefer Apple-Silicon **MPS**, then **CUDA**, then CPU. On a laptop, large models or datasets can be slow.
* Downloaded HuggingFace models are cached in `~/.cache/huggingface` and can take several GB.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `The currently activated Python version 3.13.x is not supported by the project (>=3.11,<3.13)` | Poetry picked the wrong interpreter. Install 3.12 (step 1) and run `poetry env use python3.12`. If a broken environment already exists, run `poetry env remove --all` first. |
| `The command "shell" does not exist` | Poetry 2 removed `poetry shell`. Use `poetry run ...` or `eval $(poetry env activate)`. |
| Poetry reports that the configuration in `pyproject.toml` is invalid | You are probably on Poetry 1.x. Check with `poetry --version` and upgrade to Poetry 2. |
| `litellm.AuthenticationError`, "API key not valid", "Missing ... API key" | The key is missing or wrong. Check that `.env` is in the repository root, that the variable name matches the provider (`GEMINI_API_KEY` for `gemini/...`, `OPENAI_API_KEY` for `openai/...`), and restart your kernel after editing it. |
| `sem_select` returns an empty table | Every per-column LLM call failed, often because of the API key. `sem_select` drops a column when its call fails and only prints a message. See [sem_select](operators/sem_select.md#pitfalls-and-tips). |
| `litellm.RateLimitError`, HTTP 429, "Resource has been exhausted" | You hit the provider's rate limit, which is common on the Gemini free tier. Lower the number of concurrent requests with `sempipes.update_config(batch_size_for_batch_processing=5)`, work on fewer rows (`df.head(100)`, `.skb.subsample(n=100)`), wait a minute, or switch to a paid key or another provider. |
| `UninitializedVariable: No value has been provided for 'sempipes_...'` | You passed your own environment dict to `fit`/`eval`/`cross_validate`. Start from `pipeline.skb.get_data()` instead (see [concepts.md](concepts.md#environments-always-start-from-get_data)). |
| `import sempipes` takes very long | This is expected on the first run because heavy libraries are being imported. If it is still slow after that, check that you are not on a network drive. |
| `sem_extract_features` hangs at the start | The generated code is downloading a HuggingFace model, which happens once per model. |
