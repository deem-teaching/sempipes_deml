"""Runs every numbered example script (examples/NN_*.py) end-to-end. Needs an LLM API key."""

import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT, requires_llm_api_key

EXAMPLES_DIR = REPO_ROOT / "examples"

example_files = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))


@requires_llm_api_key
@pytest.mark.parametrize("example", example_files, ids=[f.stem for f in example_files])
def test_example(example):
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{example.name} failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
