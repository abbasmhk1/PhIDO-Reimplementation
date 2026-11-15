import re
from pathlib import Path

import pytest

from phido.core.schemas import DesignerInput, VerifyInput
from phido.eval.runner import compare_templates, load_suite
from phido.pipeline import build_designer, load_pdk
from phido.stages.interpreter import EXAMPLES
from phido.stages.verify.stage import VerifyStage


def test_holdout_is_separate_and_sound():
    holdout, others = load_suite("holdout"), load_suite("all")
    assert len(holdout) == 16 and len({c.id for c in holdout}) == 16
    seen = {c.prompt.lower() for c in others} | {p.lower() for p, _ in EXAMPLES}
    assert not any(c.prompt.lower() in seen for c in holdout)  # no prompt reused from tuning or from the few-shot examples
    designer, verifier = build_designer(), VerifyStage(load_pdk())
    for c in holdout:
        gold = c.gold()
        assert compare_templates(gold, gold)[0] == "S", c.id
        d = designer.run(DesignerInput(template=gold))
        assert d.ok, (c.id, d.issues)
        assert verifier.run(VerifyInput(design=d, points=3)).ok, c.id


def test_holdout_is_not_part_of_all():
    assert not ({c.id for c in load_suite("holdout")} & {c.id for c in load_suite("all")})
