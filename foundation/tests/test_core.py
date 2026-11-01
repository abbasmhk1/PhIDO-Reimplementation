import pytest
from pydantic import ValidationError

from phido.core.schemas import EntityExtraction, PICTemplate
from phido.core.store import RunStore


def test_extraction_needs_content_and_valid_links():
    with pytest.raises(ValidationError):
        EntityExtraction()
    with pytest.raises(ValidationError):
        EntityExtraction(blocks=[{"name": "a", "kind": "mzi"}], links=[{"src": "a", "dst": "zzz"}])
    with pytest.raises(ValidationError):
        EntityExtraction(blocks=[{"name": "bad name", "kind": "mzi"}])


def test_template_validation():
    with pytest.raises(ValidationError):
        PICTemplate(nodes=[{"id": "A", "kind": "x"}], edges=[{"src": "A", "dst": "A"}])
    with pytest.raises(ValidationError):
        PICTemplate(nodes=[{"id": "A", "kind": "x"}, {"id": "A", "kind": "y"}])


def test_store_roundtrip(tmp_path):
    store = RunStore(tmp_path, "r1")
    t = PICTemplate(nodes=[{"id": "A", "kind": "mzi"}])
    store.save("x", "t", t)
    assert store.load("x", "t", PICTemplate) == t
