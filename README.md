# phido (rebuild) - interpreter, designer, layout, verify

Natural-language PIC request -> **Interpreter** -> `PICTemplate` -> **Designer** -> PIC DSL (YAML)
-> **Layout** (placement, routing, GDSII) and **Verify** (transfer functions). Layout and verify both
read the designer's DSL, independently of each other.

```
src/phido/
  core/        schemas.py (all stage contracts), stage.py, store.py (RunStore), errors.py
  llm/         adapter.py (validated objects, retry, token/time stats), mock.py, openai_compat.py
  knowledge/   pdk.py, templates.py, motifs.py, retriever.py (BM25), text.py
  geometry/    placement.py (layered placement, port alignment, crossing/overlap tests; pure Python)
  stages/      interpreter.py
               designer/  selector.py, params.py, circuit.py, proposers.py, crossings.py, stage.py
               layout/    stage.py, gds.py (gdsfactory, imported lazily)
               verify/    solver.py (S-matrix circuit solver), models.py (analytic cells), stage.py
  eval/        metrics.py, runner.py (per-stage eval), compare.py (multi-model table)
  pipeline.py  build_* helpers, run_pipeline, run_full        cli.py   `phido ...`
data/          pdk/generic_pdk.yaml, templates/templates.yaml, testbench/*_gold.yaml (basic + hard suites)
tests/         ~130 tests (layout tests need gdsfactory; SAX cross-check needs sax; both skip if absent)
```

## Design in one paragraph
Typed linear pipeline. Pydantic models are the only thing crossing stage boundaries. LLM calls go
through one adapter that validates and retries. The interpreter's LLM *only extracts entities*
(blocks/links or a motif); code builds the graph (motifs expand in code). The designer is
retrieval-first (BM25 over PDK docstrings), asks an LLM only when the choice is ambiguous, and
wires ports through validated `Connect` edits (greedy by default, LLM optional with fallback).

## Install
```bash
pip install -e ".[dev]"            # core: interpreter, designer, verify
pip install -e ".[layout]"         # adds gdsfactory (GDS output); without it layout gives an estimate only
pip install -e ".[check]"          # optional: sax, used only by a test that cross-checks our solver
```

## Use each stage on its own
```bash
phido --no-store design my_template.yaml         # designer: offline, no LLM
phido layout runs/<id>/designer/output.yaml      # layout from a stored designer DSL -> layouts/*.gds
phido verify runs/<id>/designer/output.yaml --csv spectra.csv     # transfer functions (dB per wavelength)

phido templates                                  # library templates
phido pipeline --adopt edge_mzi_edge --layout --verify   # no LLM needed: adopt a library template

export PHIDO_LLM_MODEL=qwen3:8b-8k PHIDO_NO_THINK=1       # interpreter needs an LLM
phido interpret "Two MZIs in series, each modulating up to 10 GHz"
phido pipeline "A 1x8 splitter tree from 1x2 MMIs with edge couplers at every port" --layout --verify
```

## Evaluate (each stage on its own, paper's outcome codes EE/CS/PC/SG + L/V)
```bash
phido eval designer | layout | verify            # offline, deterministic, gold templates in
PHIDO_TEMPERATURE=0.7 phido eval interpreter --suite hard --trials 5   # basic | hard | all
phido compare --suite all --models qwen3:8b-8k,qwen3:30b --trials 3   # table: pass@k, EE rate, tokens, seconds per trial; FAIL lines on stderr
```
The interpreter eval checks components, links, and any ports/parameters the gold case states.
`--suite holdout` is a separate set (16 cases) that is **not** in `--suite all`: run it once after you finish tuning, and do
not change the prompt because of its results, or it stops being held out.

## What the new stages do (and do not do)
- **Layout:** layered placement (long wires get their own track via stand-in nodes), vertical port alignment,
  staggered wire stubs, gdsfactory `route_single`, GDS. Retries with larger vertical spacing when the first attempt has
  defects. Checks (structural, not a foundry DRC): instance overlap, routing failure, routes that cross, run on top of
  each other, or pass through a component they do not connect.
  Cells without a gdsfactory equivalent (y_branch) become black boxes; pin/pn MZIs reuse the passive MZI geometry.
  The router needs vertical jogs of 0 or >= ~2 bend radii, hence `gap_y=60`; violations show up as `L` issues.
- **Verify:** in-repo S-matrix solver (numpy). Its connection logic matches SAX to ~1e-16 on an MZI netlist
  (tested when sax is installed). Cell models are analytic demo formulas (neff 2.4, ng 4.2, 2 dB/cm), checked against
  theory in the tests (cross/bar states, FSR = lambda^2/(ng dL), ring energy conservation), not foundry data.
  Warnings: light leaving through unconnected ports, outputs that receive no light, non-passive circuits.
- **State-aware wiring:** a chain element (one wire in, one out) is wired along the path that carries the light at its
  default setting, using the verify models. A 2x2 MZI at zero path difference is in the *cross* state, so it is entered
  at one input and left at the opposite-side output; with `phase: 3.1416` (bar state) the other pair is chosen. Among
  near-equal cells, a chain element prefers one that can pass light cleanly (a 2x2 switch over a 50/50 1x2). A tie that
  text matching cannot break is reported as a designer warning; name a technology or port count to decide it.

## Qwen3 / reasoning models
`<think>` blocks are stripped before JSON parsing. For speed set `PHIDO_NO_THINK=1` (appends `/no_think`).
Use a model variant with at least 8k context (e.g. `qwen3:8b-8k`): the system prompt plus schema is ~2-3k tokens.

## Extension points
- New motif: `@motif(...)` function in `knowledge/motifs.py`.
- New retriever (vectors): implement `Retriever.search(query, k, focus)`; pass to `PDK(...)` / `TemplateLibrary(...)`.
- New wiring strategy: implement `EditProposer.propose(graph, pending, feedback)`.
- Real PDK / templates / PhIDO testbench: same YAML shapes; write a small loader if the repo format differs.

## Known limits
- Designer crossing repair is a port-swap heuristic on a placement estimate (footprints), not a re-generation loop.
- Layout checks are structural; there is no real DRC deck.
- Meshes (`mesh` pattern, Clements/Reck): designer and verify work for all sizes; layout is clean for Clements meshes
  of 2-8 modes. Reck meshes of 4+ modes still get one flagged route overlap each (the checker reports it, nothing is hidden).
- Not yet run against the real PhIDO testbench/PDK/templates (format of the `PhIDO-Release` repo not seen).
- PDK `properties`, models and footprints are illustrative.
- Not yet done: PDK-size scaling study, mixed-model-per-stage experiment, vector retriever.
