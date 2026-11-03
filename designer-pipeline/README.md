# phido (rebuild) - interpreter + designer

Natural-language PIC request -> **Interpreter** -> `PICTemplate` -> **Designer** -> PIC DSL (YAML).
Layout and verification stages come later and plug in as more `Stage` classes.

Based on: A. Sharma, Y. Fu, V. Ansari, *et al.*, "AI agents for photonic integrated circuit design automation",
*APL Mach. Learn.* **3**, 046113 (2025), [doi:10.1063/5.0300741](https://doi.org/10.1063/5.0300741).
This is an independent reimplementation, not the authors' code (theirs: <https://github.com/JPPhotonics/PhIDORelease>).

```
src/phido/
  core/        schemas.py (all stage contracts), stage.py (Stage base), store.py (RunStore), errors.py
  llm/         adapter.py (generate -> validated object, retry), mock.py, openai_compat.py
  knowledge/   pdk.py, templates.py, motifs.py, retriever.py (Retriever protocol + BM25), text.py
  stages/      interpreter.py
               designer/  selector.py, params.py, circuit.py, proposers.py, stage.py
  eval/        metrics.py (pass@k, absolute/conditional rates), runner.py (per-stage eval)
  pipeline.py  build_* helpers + run_pipeline        cli.py   `phido ...`
data/          pdk/generic_pdk.yaml, templates/templates.yaml, testbench/*_gold.yaml
examples/      my_template.yaml
tests/         no network, no real LLM
```

## Design in one paragraph
Typed linear pipeline. Pydantic models are the only thing crossing stage boundaries. LLM calls go
through one adapter that validates and retries. The interpreter's LLM *only extracts entities*
(blocks/links or a motif); code builds the graph (motifs expand in code). The designer is
retrieval-first (BM25 over PDK docstrings), asks an LLM only when the choice is ambiguous, and
wires ports through validated `Connect` edits (greedy by default, LLM optional with fallback).

## Use each stage on its own
```bash
pip install -e ".[dev]" && pytest

# Designer: offline, deterministic, no LLM
phido design examples/my_template.yaml   # PICTemplate YAML (or an interpreter output.yaml)
phido eval designer                      # gold templates -> expected cells, per-stage outcome codes

# Interpreter: needs an LLM (any OpenAI-compatible endpoint; Ollama by default)
export PHIDO_LLM_MODEL=qwen2.5:3b        # PHIDO_LLM_BASE_URL / PHIDO_LLM_API_KEY / PHIDO_TEMPERATURE optional
phido interpret "Two MZIs in series, each modulating up to 10 GHz"
phido eval interpreter --trials 5        # use PHIDO_TEMPERATURE>0 for meaningful pass@5

phido pipeline "..." [--llm-design]      # chained; artifacts under runs/<id>/<stage>/
```
In Python: `InterpreterStage(llm, library).run(InterpreterInput(prompt=...))` and
`DesignerStage(pdk).run(DesignerInput(template=...))` are independent; the designer can be
re-run from a stored `runs/<id>/interpreter/output.yaml`.

## Qwen3 / reasoning models
`<think>` blocks are stripped before JSON parsing. For speed set `PHIDO_NO_THINK=1` (appends `/no_think`).
Use a model variant with at least 8k context (e.g. `qwen3:8b-8k`): the system prompt plus schema is ~2-3k tokens.

## Extension points
- New motif: `@motif(...)` function in `knowledge/motifs.py`.
- New retriever (vectors): implement `Retriever.search(query, k, focus)`; pass to `PDK(...)` / `TemplateLibrary(...)`.
- New wiring strategy: implement `EditProposer.propose(graph, pending, feedback)`.
- Real PDK / templates / PhIDO testbench: same YAML shapes; write a small loader if the repo format differs.

## Known limits (v0)
- Planarity check is graph-level (warns); port-order heuristics avoid crossings for trees/chains only.
- Interpreter eval compares component kinds + connectivity, not ports or params yet.
- PDK `properties` values are illustrative, not foundry data.
- Not yet tested against a real small LLM; prompt/few-shot may need tuning for your model.

## License
MIT, see [LICENSE](LICENSE).
