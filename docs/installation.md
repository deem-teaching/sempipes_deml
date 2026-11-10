# Installation

This page gets you from a fresh machine to a working `import sempipes`. Expect the first install to take a while: SemPipes pulls in PyTorch, HuggingFace `transformers`, AutoGluon and SDV.

## 1. Python 3.11 or 3.12

SemPipes requires **Python 3.11 or 3.12** (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`). Python 3.13 is **not** supported, and many systems now ship it as their default, so check first:

```bash
python3 --version
python3.12 --version   # is a 3.12 interpreter available?
```

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
poetry install               # installs SemPipes + all dependencies (several GB, can take a while)
```

## 4. API keys

Put your key into a `.env` file in the repository root:

```bash
cp .env.example .env
```

Then edit `.env` and fill in at least one key:

```bash
GEMINI_API_KEY=...   # needed for the default model (gemini/gemini-2.5-flash)
OPENAI_API_KEY=...   # only if you switch to an openai/... model
HF_TOKEN=...         # optional, required for sem_extract_)features
```

* **SAIA (GWDG Academic Cloud):** free for members of German universities. Put the key and the base URL into `.env` as `SAIA_API_KEY` and `SAIA_BASE_URL` and switch with `sempipes._set_config(sempipes.get_default_saia_config())`, or make SAIA the default for everything (including the tests) with `SEMPIPES_LLM_PROVIDER=saia` in `.env` (see [configuration.md](configuration.md#using-saia-gwdg-academic-cloud)).

* **Gemini:** you can create a free API key in [Google AI Studio](https://aistudio.google.com/apikey but be careful, it can be expensive). The free tier has rate limits, which matter for operators that send one request per row (see [Troubleshooting](#troubleshooting)).

* **`HF_TOKEN` (optional):** `sem_extract_features` in its default code mode runs HuggingFace models locally and downloads them from the HuggingFace Hub on first use. A token is only needed for gated models, and it also raises your download rate limits.

* **Local models via [Ollama](https://ollama.com/):** these need no key. Install Ollama, pull a model (e.g. `ollama pull gpt-oss:20b`) and point SemPipes at it with `api_base` (see [configuration.md](configuration.md#provider-examples)). Small local models often struggle to write working code, so expect more retries.

`import sempipes` calls `load_dotenv()`, which finds the `.env` file in the repository root automatically as long as you work inside the repository folder (scripts, notebooks in `examples/notebooks/`, `poetry run python -c ...` from the root).


## 5. Verify the installation

```bash
poetry run python -c "import sempipes; print(sempipes.get_config())"
```

The first import is slow. `import sempipes` loads torch, transformers, AutoGluon and SDV because generated code may use them. Later imports are faster.

To check that your API key works (this makes one small LLM call):

```bash
poetry run python -c "import sempipes, litellm; r = litellm.completion(model=sempipes.get_config().llm_for_code_generation.name, messages=[{'role': 'user', 'content': 'Say hi'}]); print(r.choices[0].message.content)"
```

## 6. Running code

Run scripts through Poetry from the repository root:

```bash
poetry run python examples/01_sem_select.py
```

Alternatively, activate the environment in your shell. 

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
