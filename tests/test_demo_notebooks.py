"""Executes the demo notebooks in examples/notebooks/. Needs an LLM API key."""

import nbformat
import pytest
from nbclient import NotebookClient

from tests.conftest import REPO_ROOT, requires_llm_api_key

NOTEBOOKS_DIR = REPO_ROOT / "examples" / "notebooks"

NOTEBOOKS = [
    "demo.ipynb",
    "demo__sem_fillna.ipynb",
    "demo__sem_select.ipynb",
    "demo__sem_agg_features.ipynb",
    "demo__optimise_semantic_operator.ipynb",
    "demo__optimise_multiple_semantic_operators.ipynb",
]


@requires_llm_api_key
@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_demo_notebook(notebook):
    with open(NOTEBOOKS_DIR / notebook, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    # Execute with the notebook folder as working directory, like a student would
    client = NotebookClient(nb, resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    client.execute()
