import pytest
from pydantic import BaseModel

from phido.llm.adapter import LLMFormatError, extract_json
from conftest import make_llm


class Thing(BaseModel):
    n: int


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"n": 1}\n```') == {"n": 1}
    assert extract_json('noise {"n": 2} trailing') == {"n": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_retries_with_feedback_then_succeeds():
    llm, mock = make_llm(["garbage", '{"n": "x"}', '{"n": 3}'])
    assert llm.generate("sys", "user", Thing).n == 3
    assert len(mock.calls) == 3
    assert "rejected" in mock.calls[1][1]
    trace = llm.drain_trace()
    assert [t.get("ok", False) for t in trace] == [False, False, True]
    assert llm.drain_trace() == []


def test_validate_hook_triggers_retry():
    llm, _ = make_llm(['{"n": 1}', '{"n": 5}'])

    def check(t):
        if t.n < 5:
            raise ValueError("too small")

    assert llm.generate("s", "u", Thing, check).n == 5


def test_gives_up_after_budget():
    llm, mock = make_llm(["bad"] * 5, max_retries=1)
    with pytest.raises(LLMFormatError):
        llm.generate("s", "u", Thing)
    assert len(mock.calls) == 2


def test_extract_json_ignores_think_block_with_braces():
    raw = '<think>maybe {"n": 99} or {x}?</think>\n{"n": 7}'
    assert extract_json(raw) == {"n": 7}
