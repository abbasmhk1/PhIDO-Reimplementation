from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class RunStore:
    """Every stage writes input/output here: runs/<run_id>/<stage>/<name>.yaml.

    Lets you inspect, hand-edit and resume from any stage, and replay evals.
    """

    def __init__(self, root: str | Path = "runs", run_id: str | None = None):
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = Path(root) / self.run_id

    def path(self, stage: str, filename: str) -> Path:
        p = self.dir / stage / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, stage: str, name: str, obj: BaseModel) -> Path:
        p = self.path(stage, f"{name}.yaml")
        p.write_text(
            yaml.safe_dump(obj.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return p

    def save_text(self, stage: str, filename: str, text: str) -> Path:
        p = self.path(stage, filename)
        p.write_text(text, encoding="utf-8")
        return p

    def save_json(self, stage: str, filename: str, data: Any) -> Path:
        return self.save_text(stage, filename, json.dumps(data, indent=2, ensure_ascii=False))

    def load(self, stage: str, name: str, model: type[T]) -> T:
        p = self.dir / stage / f"{name}.yaml"
        return model.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
