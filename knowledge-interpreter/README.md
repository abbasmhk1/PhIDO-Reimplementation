# phido (rebuild)

LLM-assisted photonic IC design pipeline, rebuilt in typed, independently testable stages:
natural-language PIC request -> **Interpreter** -> `PICTemplate` -> **Designer** -> PIC DSL (YAML).
Layout and verification stages come later as more `Stage` classes.

Based on: A. Sharma, Y. Fu, V. Ansari, *et al.*, "AI agents for photonic integrated circuit design automation",
*APL Mach. Learn.* **3**, 046113 (2025), [doi:10.1063/5.0300741](https://doi.org/10.1063/5.0300741).
This is an independent reimplementation, not the authors' code (theirs: <https://github.com/JPPhotonics/PhIDORelease>).

## Status
The repo is being published in three steps.

- [x] 1. Foundation - stage contracts (pydantic schemas), `Stage` base, `RunStore`, LLM adapter
- [x] **2. Knowledge + interpreter** - PDK/template/motif knowledge, BM25 retriever, `InterpreterStage`
- [ ] 3. Designer, pipeline, eval and CLI

## What is here
```
src/phido/
  core/        schemas.py, stage.py, store.py, errors.py
  llm/         adapter.py, mock.py, openai_compat.py
  knowledge/   pdk.py, templates.py, motifs.py, retriever.py (Retriever protocol + BM25), text.py
  stages/      interpreter.py
data/          pdk/generic_pdk.yaml, templates/templates.yaml
tests/         no network, no real LLM
```

The interpreter's LLM *only extracts entities* (blocks/links or a motif name); code builds the graph, and motifs expand
in code. Templates are retrieved with BM25 over a curated library.

```python
from phido.core.schemas import InterpreterInput
from phido.knowledge.templates import TemplateLibrary
from phido.stages.interpreter import InterpreterStage

library = TemplateLibrary.load("data/templates/templates.yaml")
out = InterpreterStage(llm, library).run(InterpreterInput(prompt="Two MZIs in series"))
```
`llm` is a `StructuredLLM` wrapping any OpenAI-compatible client (see `phido/llm/openai_compat.py`).

## Install and test
```bash
pip install -e ".[dev]" && pytest
```

## Extension points
- New motif: `@motif(...)` function in `knowledge/motifs.py`.
- New retriever (vectors): implement `Retriever.search(query, k, focus)`; pass to `PDK(...)` / `TemplateLibrary(...)`.

## License
MIT, see [LICENSE](LICENSE).
