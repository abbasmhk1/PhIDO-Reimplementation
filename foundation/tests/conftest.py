import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phido.llm.adapter import StructuredLLM  # noqa: E402
from phido.llm.mock import MockLLM  # noqa: E402


def make_llm(responses, **kw):
    mock = MockLLM(responses)
    return StructuredLLM(mock, **kw), mock


@pytest.fixture
def llm_factory():
    return make_llm


def j(obj) -> str:
    return json.dumps(obj)
