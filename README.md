# phido

**Plain-English photonic circuit design.** Describe a photonic integrated circuit in a sentence; phido turns it into a
circuit graph, picks components from a PDK, wires the ports, lays it out as a GDSII file, and simulates its
transmission spectrum.

```text
"A 1x4 power splitter made of 1x2 MMIs, with an edge coupler at the input
 and a photodetector at every output."
        │
        ▼  Interpreter ─ LLM extracts the structure, code builds the graph
        ▼  Designer ───  picks PDK cells, parameters and port-level wiring
        ├─▶ Layout ────  placement, routing, structural checks, GDSII
        └─▶ Verify ────  circuit simulation: transmission vs wavelength
```

phido is an independent re-implementation of the pipeline in *"AI agents for photonic integrated circuit design
automation"* (Sharma et al., APL Machine Learning 3, 046113 (2025), doi:10.1063/5.0300741). It is not affiliated with the
authors and does not use their code or data. See [How it differs from the paper](#how-it-differs-from-the-paper).

> **Status:** a working prototype with a demo PDK. Every stage runs and is tested, but the cell physics uses
> illustrative constants, not foundry data. See [Limitations](#limitations).

## Contents
[Quick start](#quick-start) · [Example](#example) · [How it works](#how-it-works) · [Configuration](#configuration) ·
[Commands](#commands) · [Evaluation](#evaluation) · [Results so far](#results-so-far) · [Project layout](#project-layout) ·
[Extending](#extending-phido) · [Limitations](#limitations) · [Troubleshooting](#troubleshooting)

## Quick start
Python 3.10 or newer.

```bash
pip install -e ".[dev,layout]"      # core + pytest + gdsfactory (for GDS output)
pytest -q                           # offline, no LLM needed
```

Try it **without any LLM**: adopt a template from the built-in library, then design, lay out and simulate it.
```bash
phido templates                                           # list library templates
phido pipeline --adopt edge_mzi_edge --layout --verify
```

With an LLM (any OpenAI-compatible endpoint, Ollama by default):
```bash
export PHIDO_LLM_MODEL=qwen3:8b-8k            # see Configuration for the other variables
phido pipeline "An edge coupler feeds a 2x2 MZI, whose output goes to a photodetector." --layout --verify
```
Results go to `runs/<id>/`: the design in `designer/output.yaml`, the layout in `layout/*.gds` (open with KLayout), and the
spectra in `verify/output.yaml`.

## Example
```console
# output abridged (the real design is full YAML, one entry per node and edge)
$ phido pipeline "A 1x4 power splitter made of 1x2 MMIs, with an edge coupler at the input and a photodetector at every output." --layout --verify
doc:
  title: 2-ary tree of mmi, 2 stages
nodes: {N1: mmi_1x2, N2: mmi_1x2, N3: mmi_1x2, IN1: edge_coupler, OUT1..OUT4: photodetector}
edges: {E1: "N1,o2:N2,o1", E2: "N1,o3:N3,o1", E3: "IN1,o1:N1,o1", E4: "N2,o2:OUT1,o1", ...}

layout: 8 components, 7/7 routes ok, GDS: runs/.../layout/2_ary_tree_of_mmi_2_stages.gds
verify: inputs=['IN1,fiber'] lit outputs=4 unconnected ports=0 (0 lit)
  IN1,fiber -> OUT1,pd:    -8.12 dB at centre, peak -8.12 dB @ 1.5 um, min -8.12 dB
  ... (all four outputs identical)
```
The −8.12 dB is the expected budget: 1/4 split (6.02 dB) + two MMIs at 0.3 dB + the 1.5 dB input coupler.

More prompts to try:

| Prompt | What you should see |
|---|---|
| `An edge coupler feeds a 2x2 MZI, whose output goes to a photodetector.` | −1.52 dB, no warnings |
| `An edge coupler feeds a 2x2 MZI with a path length difference of 100 um, and the MZI output goes to a photodetector.` | fringes every 5.72 nm (λ²/(n_g·ΔL)); a warning that the complementary light leaves through the unconnected port |
| `An edge coupler feeds an add-drop ring resonator with 2 inputs and 2 outputs; the through port goes to a photodetector.` | resonance notches every ~9.1 nm; a warning about the drop port |
| `A 4-mode Clements mesh of MZIs with an edge coupler at every input and output.` | 14 components, 16/16 routes, −3.06 dB per input |

To plot a spectrum (use `--points 2001` or more for narrow features such as ring resonances):
```bash
phido --no-store verify runs/<id>/designer/output.yaml --points 2001 --csv spectrum.csv
```

**Writing good prompts:** name the parts (MZI, MMI, ring resonator, edge/grating coupler, photodetector, heater,
directional coupler, straight waveguide); give counts and ports (`2x2`, `1x2`); give numbers with units
(`radius 12 um`, `path length difference 100 um`, `30 GHz`); say where couplers go (`at the input only`,
`at every output`); name a technology for MZIs (`thermo-optic`, `pin-diode`) to avoid an "ambiguous choice" warning.
Repeating structures are understood as patterns: trees, cascades, and Clements/Reck meshes.

## How it works
```text
 prompt ──▶ INTERPRETER ──PICTemplate──▶ DESIGNER ──PIC DSL──┬──▶ LAYOUT ──▶ GDSII
            (LLM + code)                  (retrieval + code)  └──▶ VERIFY ──▶ spectra
                │                              │
         circuit templates               PDK (cells, ports,
         + motif generators              parameters, models)

 every stage reads and writes validated pydantic models and saves to runs/<id>/<stage>/
```

- **Interpreter.** The LLM only *extracts entities*: blocks, links, or a named pattern (`tree`, `cascade`, `mesh`). Code
  builds the graph, so a 63-component tree costs a few output tokens and cannot be miscounted. A retriever suggests
  similar library templates. Output retries with validation feedback if the model returns bad JSON.
- **Designer.** BM25 retrieval over PDK docstrings ranks cells; an LLM is consulted only to break near-ties. Parameters are
  mapped through aliases and range-checked. Wiring uses validated `Connect` edits: components are placed left to
  right, ports are assigned to avoid crossings, and a chain element (one wire in, one out) is wired along the path that
  actually carries light in its default state. Anything the designer is unsure about is reported as a warning.
- **Layout.** Layered placement with stand-in tracks for long wires, port alignment, staggered wire stubs,
  gdsfactory routing, and structural checks (overlaps, routing failures, wires that cross, overlap, or pass through
  another component). Retries with more spacing if the first attempt has defects.
- **Verify.** A scattering-matrix circuit solver plus analytic cell models gives transmission between external ports over a
  wavelength sweep, with warnings for light leaving through unconnected ports, dark outputs, and non-passive circuits.

## Configuration
Environment variables for the LLM (the interpreter needs one; the rest of the pipeline does not):

| Variable | Meaning | Default |
|---|---|---|
| `PHIDO_LLM_MODEL` | model name | *(required for `interpret`, `eval`, `pipeline`)* |
| `PHIDO_LLM_BASE_URL` | OpenAI-compatible endpoint | `http://localhost:11434/v1` (Ollama) |
| `PHIDO_LLM_API_KEY` | API key if the endpoint needs one | none |
| `PHIDO_TEMPERATURE` | sampling temperature; use `0.7` for `--trials` above 1 | `0` |
| `PHIDO_NO_THINK` | `1` appends Qwen3's `/no_think` switch for faster extraction | off |
| `PHIDO_DATA` | alternative data folder (PDK, templates, test cases) | `./data` |

Global options go before the command: `--runs-dir DIR`, `--no-store` (do not write run artifacts).

**Models.** The interpreter prompt plus schema is about 2–3k tokens, so use a model with at least an 8k context.
`<think>` blocks from reasoning models are stripped before parsing. A model on a remote server works through an SSH
tunnel: `ssh -N -L 11435:HOST:11434 user@gateway`, then `PHIDO_LLM_BASE_URL=http://localhost:11435/v1`.

## Commands
```bash
phido interpret "PROMPT"                      # prompt -> circuit template        (LLM)
phido design template.yaml [--llm]            # template -> PIC DSL               (offline; --llm for ambiguity/wiring help)
phido layout designer_output.yaml             # DSL -> placement, routing, GDS    (needs gdsfactory)
phido verify designer_output.yaml [--csv F] [--wl START STOP] [--points N]
phido pipeline "PROMPT" [--layout] [--verify] [--llm-design]
phido pipeline --adopt NAME [--layout] [--verify]       # library template, no LLM
phido templates                               # list library templates
phido eval designer|layout|verify             # offline stage evaluations
phido eval interpreter [--suite S] [--trials N]
phido compare --models a,b [--suite S] [--trials N]     # accuracy, tokens and time per model
```
A run folder looks like `runs/<id>/{interpreter,designer,layout,verify}/` with `input.yaml`, `output.yaml`,
`llm_trace.json` (every LLM attempt, with tokens and seconds) and `error.txt` when a stage fails. Any stage can be
re-run from a stored file, and you can edit an `output.yaml` by hand before the next stage.

## Evaluation
Each stage is tested **on its own**, with gold inputs, so an upstream mistake cannot hide a downstream result. Outcome
codes follow the paper: `EE` entity extraction, `CS` component selection, `PC` parameter configuration, `SG` schematic
generation; plus `L` layout and `V` verification. Reported metrics: pass@k, absolute and conditional occurrence rates, and for
LLM stages, calls, tokens and seconds per trial.

| Suite | Cases | Purpose |
|---|---|---|
| `basic` | 13 | single components, short chains, simple trees |
| `hard` | 18 | hierarchy, parts-versus-components, per-node parameters, patterns, meshes |
| `all` | 31 | `basic` + `hard` (the prompt was tuned on these) |
| `holdout` | 16 | written after tuning, **not** in `all`; run once, do not tune against it |

The interpreter score checks components, connectivity, and any ports or parameters a gold case states. Parameter names
match through the PDK's alias table (`coupling_length_um` = `length_um`).

## Results so far
Measured on the author's setup with local Ollama models; these are **my own benchmarks, not the paper's**, so the numbers
are not comparable to its tables.

| Check | Result |
|---|---|
| Unit and integration tests | 136 passing, offline |
| Interpreter, `basic` + `hard` (31 cases × 2 trials), `qwen3.8:27b-8k` | 62/62 (prompt tuned on these) |
| Interpreter, `holdout` (16 cases × 2 trials) | 16/16 for two models, after fixing one scorer bug (15/16 before; the prompt was not changed) |
| Designer with an LLM chooser and wiring proposer, 9 gold cases × 2 | 9/9 |
| Layout, Clements meshes of 2–8 modes (up to 44 components) | clean |
| Verify vs theory | MZI fringe spacing 5.72 nm (theory 5.72 nm), ring notch spacing 9.14 nm (theory 9.10 nm); circuit solver agrees with SAX to ~1e-16 |

The benchmarks are small (tens of trials per model), so differences between strong models are not statistically
meaningful. Their value is in finding systematic errors, which they did: a missing input-only/output-only coupler option, null
handling, hierarchy counting, and a wiring flaw that sent light to unused ports.

## Project layout
```text
src/phido/
  core/        schemas.py (all stage contracts), stage.py, store.py (run folders), errors.py
  llm/         adapter.py (validated objects, retry, token/time stats), mock.py, openai_compat.py
  knowledge/   pdk.py, templates.py, motifs.py (tree/cascade/mesh), retriever.py (BM25), text.py
  geometry/    placement.py (layering, port alignment, crossing/overlap tests; pure Python)
  stages/      interpreter.py
               designer/  selector.py, params.py, circuit.py, proposers.py, crossings.py, throughput.py, stage.py
               layout/    stage.py, gds.py (gdsfactory, imported lazily)
               verify/    solver.py (S-matrix solver), models.py (analytic cells), stage.py
  eval/        metrics.py, runner.py, compare.py
  pipeline.py  cli.py
data/          pdk/generic_pdk.yaml, templates/templates.yaml, testbench/*.yaml
tests/
```

## Extending phido
- **New pattern:** add a `@motif(...)` function in `knowledge/motifs.py`; the interpreter prompt lists registered patterns automatically.
- **New cells:** add them to `data/pdk/generic_pdk.yaml` (ports as `"LxR"`, parameters with ranges and aliases, a `model`
  name, a `layout` entry for gdsfactory). Add the model to `stages/verify/models.py`.
- **New retriever** (e.g. vectors for a large PDK): implement `Retriever.search(query, k, focus)` and pass it to `PDK(...)`.
- **New wiring strategy:** implement `EditProposer.propose(graph, pending, feedback)`.
- **Your own data:** point `PHIDO_DATA` at a folder with the same YAML shapes.
- **New stage:** subclass `Stage` with pydantic input and output models; add it in `pipeline.py`.

## How it differs from the paper
| Paper | phido |
|---|---|
| Two LLM agents (interpreter, designer) | The LLM does the minimum: extraction in the interpreter, tie-breaks in the designer |
| LLM generates the circuit schematic and port wiring | Code builds graphs from extracted entities and patterns; wiring uses placement geometry and cell physics |
| Schema checks routed through GPT-4o for non-OpenAI models | One adapter validates and retries for every model |
| Layout via Graphviz placement and gdsfactory routing | Own layered placement plus gdsfactory routing, with checks for overlaps and pass-through wires |
| Verification with SAX | Own S-matrix solver (numpy only), cross-checked against SAX |
| 34-cell generic PDK, 102-prompt testbench | 18-cell demo PDK, 47 interpreter cases written for this project |

## Limitations
- **Demo data.** The PDK, cell models (n_eff 2.4, n_g 4.2, 2 dB/cm, ring coupling) and footprints are illustrative. Predictions
  are only as good as the numbers; replace them with your PDK's data before trusting absolute values.
- **Layout checks are structural.** There is no foundry design-rule deck. Reck meshes of 4+ modes get one flagged route
  overlap each (reported, not hidden). `y_branch` is a black box; pin/pn MZIs reuse the passive MZI geometry.
- **Not run on the paper's data.** The paper's testbench, PDK and templates were not used.
- **Designer crossing repair** is a port-swap heuristic on a placement estimate, not an LLM regeneration loop.
- **Not built:** the paper's PDK-size scaling study, a per-stage mixed-model study, a vector retriever.

## Troubleshooting
| Symptom | Cause and fix |
|---|---|
| `error: Set PHIDO_LLM_MODEL ...` | The variable is not set in this shell. Variables do not carry over to new terminals; keep them in a file and `source` it. |
| `LLM endpoint ... HTTP Error 404` | Wrong base URL or model name: the default `localhost:11434` is your local Ollama. Set `PHIDO_LLM_BASE_URL` and check the model exists with `curl $PHIDO_LLM_BASE_URL/../api/tags` (Ollama). |
| Empty or truncated model answers | Context too small; use a model variant with at least 8k tokens. |
| `<think>` text or very slow answers | Reasoning model; set `PHIDO_NO_THINK=1` (Qwen3) or choose a non-thinking model. |
| `gdsfactory is not installed` warning | `pip install -e ".[layout]"`; without it layout only estimates placement. |
| `... matched several cells about equally` | Not an error: the prompt did not say which MZI/ring variant. Name a technology or port count. |
| `light ... leaves through unconnected port ...` | Real physics: the component sends part of the light to an output you did not connect (filters, rings, splitters). |
| Ring notches look shallow | The sweep is too coarse. Use `--points 2001` or `--wl START STOP` to zoom. |
| `phido: command not found` | Activate the environment where you ran `pip install -e .`. |

## Citation and credit
If this is useful, please cite the paper it re-implements:
Sharma, A., Fu, Y., Ansari, V., Iyer, R., Kuang, F., Mistry, K., Aishy, R. I., Ahmad, S., Matres, J., Englund, D. R.,
Poon, J. K. S. *AI agents for photonic integrated circuit design automation.* APL Machine Learning **3**, 046113 (2025).
https://doi.org/10.1063/5.0300741. The authors' reference implementation is the open-source PhIDO-Release repository.

## License
*(Choose one before publishing, e.g. MIT or Apache-2.0, and add a `LICENSE` file.)*
