# PhIDO-Reimplementation

**An independent Python reimplementation of an LLM-assisted photonic integrated circuit (PIC) design workflow.**

PhIDO-Reimplementation explores how natural-language PIC requirements can be translated into structured circuit representations and then mapped to photonic components using a modular, typed software architecture.

The project is inspired by the research paper:

> A. Sharma, Y. Fu, V. Ansari, *et al.*, “AI agents for photonic integrated circuit design automation,” *APL Machine Learning*, **3**, 046113 (2025).  
> https://doi.org/10.1063/5.0300741

**This repository is an independent reimplementation. It is not the original authors’ code and is not affiliated with the original authors.** The original project is available at [JPPhotonics/PhIDORelease](https://github.com/JPPhotonics/PhIDORelease).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Implemented Features](#implemented-features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Run the Designer](#1-run-the-designer)
  - [Run the Interpreter](#2-run-the-interpreter)
  - [Run the Full Pipeline](#3-run-the-full-pipeline)
  - [Run Evaluation](#4-run-evaluation)
  - [Use the Stages from Python](#5-use-the-stages-from-python)
- [Design Principles](#design-principles)
- [Extending the Project](#extending-the-project)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

Designing a photonic integrated circuit involves translating functional requirements into a circuit topology, selecting suitable components, and connecting their optical ports. This project investigates how LLM-assisted agents and conventional programmatic methods can support parts of that workflow.

The implementation is organized as a sequence of independently executable stages:

```text
Natural-language PIC request
             |
             v
        Interpreter
             |
             v
         PICTemplate
             |
             v
          Designer
             |
             v
        PIC DSL (YAML)
```

The **Interpreter** converts a natural-language description into a structured `PICTemplate`. The **Designer** uses the structured template and available PDK knowledge to select components and construct a circuit representation, producing a PIC DSL representation in YAML.

The architecture is stage-based: each stage receives a typed input, performs its task, and returns a typed output. This makes individual stages easier to test, inspect, and extend.

> **Scope:** This repository implements the Interpreter and Designer foundations, along with shared infrastructure, knowledge and retrieval utilities, evaluation support, and command-line/pipeline entry points. Physical layout generation and full physical verification are not implemented as complete downstream stages in this release.

---

## Architecture

### 1. Interpreter

The Interpreter processes a natural-language PIC request and produces a structured `PICTemplate`.

Its LLM is used primarily for extracting entities from the request, such as:
- Photonic component or block types.
- Connections and relationships between blocks.
- A recognized circuit motif, where applicable.

The circuit graph is constructed programmatically. Motif expansion is also handled in code rather than relying on the LLM to generate the entire graph directly. The Interpreter can retrieve relevant examples from the template library using BM25 retrieval.

### 2. Designer

The Designer transforms a `PICTemplate` into a component-level circuit representation.

Its approach is retrieval-first:
- It searches the available PDK knowledge, including component descriptions.
- It uses programmatic selection and parameter-handling utilities.
- When a component choice is ambiguous, an LLM-assisted proposal can be used.
- Port connections are represented through validated `Connect` edits.
- The default wiring approach is greedy; an LLM-based proposer is optional and includes a fallback path.

The Designer can also be run independently from an existing `PICTemplate`, without running the Interpreter first.

### 3. Shared infrastructure

The stages use shared Pydantic contracts and a common stage abstraction. LLM calls are routed through a structured adapter that validates generated outputs and supports retries. A mock backend supports tests that do not require a live model or network access.

---

## Implemented Features

### Foundation

- **Typed data contracts:** Pydantic schemas define structured inputs and outputs for processing stages.
- **Stage abstraction:** A shared interface supports modular, independently executable stages.
- **Run storage and errors:** Utilities support run artifacts and stage-related error handling.
- **Structured LLM adapter:** LLM output is validated against the expected data model, with retry support.
- **LLM backends:** Includes a mock backend and an OpenAI-compatible client adapter.

### Knowledge and retrieval

- **PDK knowledge utilities:** Access to photonic component information.
- **Template library:** YAML-based circuit templates.
- **Motif library:** Reusable circuit patterns, with programmatic motif expansion.
- **BM25 retrieval:** Search over the available template/PDK knowledge.
- **Retriever protocol:** Defines an interface that can be implemented by alternative retrieval methods.
- **Text utilities:** Supporting text-processing functionality.

### Designer

- Retrieval-first component selection using PDK descriptions.
- Parameter-handling utilities.
- Circuit construction and port-connection logic.
- Greedy wiring proposal by default.
- Optional LLM-assisted edit proposal with fallback behavior.

### Evaluation and execution

- Per-stage evaluation support.
- Metrics including `pass@k` and absolute/conditional outcome rates.
- Command-line entry points for running individual stages, evaluation, and the chained pipeline.
- Run artifacts organized by run and stage.

---

## Repository Structure

The main source and data organization is:

```text
src/
└── phido/
    ├── core/
    │   ├── schemas.py       # Typed contracts for stage inputs and outputs
    │   ├── stage.py         # Shared Stage abstraction
    │   ├── store.py         # RunStore and run-artifact support
    │   └── errors.py        # Project-specific errors
    │
    ├── llm/
    │   ├── adapter.py       # Structured generation, validation, and retry
    │   ├── mock.py          # Mock LLM backend
    │   └── openai_compat.py # OpenAI-compatible client adapter
    │
    ├── knowledge/
    │   ├── pdk.py           # PDK knowledge utilities
    │   ├── templates.py     # Template library
    │   ├── motifs.py        # Circuit motifs and expansion
    │   ├── retriever.py     # Retriever protocol and BM25 retrieval
    │   └── text.py          # Text-processing utilities
    │
    ├── stages/
    │   ├── interpreter.py   # Natural-language to PICTemplate
    │   └── designer/
    │       ├── selector.py
    │       ├── params.py
    │       ├── circuit.py
    │       ├── proposers.py
    │       └── stage.py
    │
    ├── eval/
    │   ├── metrics.py       # Evaluation metrics
    │   └── runner.py        # Per-stage evaluation runner
    │
    ├── pipeline.py          # Pipeline construction and execution
    └── cli.py               # Command-line interface

data/
├── pdk/
│   └── generic_pdk.yaml
├── templates/
│   └── templates.yaml
└── testbench/
    └── *_gold.yaml

examples/
└── my_template.yaml

tests/
    # Offline tests; no live LLM required

pyproject.toml
README.md
LICENSE
```

---

## Requirements

- Python **3.10 or newer**.
- The Python packages declared in `pyproject.toml`:
  - `pydantic>=2`
  - `pyyaml`
  - `networkx`
- `pytest` for running the development test suite.
- A compatible LLM endpoint for Interpreter runs. The Designer can be run without a live LLM using its default deterministic workflow.

---

## Installation

Clone the repository and move into its root directory:

```bash
git clone <YOUR-REPOSITORY-URL>
cd PhIDO-Reimplementation
```

Install the project in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

The included tests are intended to run without network access or live LLM calls.

---

## Configuration

The Interpreter requires an LLM backend. The project provides an OpenAI-compatible adapter; the README's example configuration uses an Ollama-compatible local endpoint.

Set the model name before running the Interpreter:

```bash
export PHIDO_LLM_MODEL=qwen2.5:3b
```

Optional environment variables:

| Variable | Purpose |
|---|---|
| `PHIDO_LLM_MODEL` | Model identifier used by the LLM adapter. |
| `PHIDO_LLM_BASE_URL` | Base URL of the OpenAI-compatible endpoint. |
| `PHIDO_LLM_API_KEY` | API key, when required by the endpoint. |
| `PHIDO_TEMPERATURE` | Controls generation temperature. |
| `PHIDO_NO_THINK` | Enables the `/no_think` suffix behavior for supported reasoning-model workflows. |

The exact endpoint and model availability depend on the LLM provider or local runtime being used.

### Reasoning models and Qwen

For reasoning models that emit `<think>` blocks, the implementation strips those blocks before JSON parsing. To request a faster non-thinking mode where supported, set:

```bash
export PHIDO_NO_THINK=1
```

The original project notes recommend using a model variant with at least an 8k context window (for example, `qwen3:8b-8k`). The system prompt and schema are estimated in the project notes to require approximately 2–3k tokens.

---

## Usage

The package exposes a `phido` command-line entry point.

### 1. Run the Designer

The Designer can be executed independently and does not require a live LLM in its default deterministic mode.

```bash
phido design examples/my_template.yaml
```

The input should be a `PICTemplate` YAML file, such as the example template or a saved Interpreter output.

### 2. Run the Interpreter

The Interpreter accepts a natural-language PIC request and requires a configured LLM endpoint.

```bash
phido interpret "Two MZIs in series, each modulating up to 10 GHz"
```

### 3. Run the full pipeline

The pipeline chains the stages:

```bash
phido pipeline "Two MZIs in series, each modulating up to 10 GHz"
```

An optional flag enables LLM-assisted Designer behavior:

```bash
phido pipeline "Two MZIs in series, each modulating up to 10 GHz" --llm-design
```

Pipeline artifacts are stored under a run-specific directory, following the structure:

```text
runs/
└── <run-id>/
    ├── interpreter/
    └── designer/
```

The Designer can be re-run separately from a stored Interpreter output, allowing the two stages to be inspected and executed independently.

### 4. Run evaluation

Evaluate the Designer:

```bash
phido eval designer
```

Evaluate the Interpreter over multiple trials:

```bash
phido eval interpreter --trials 5
```

For a meaningful `pass@k` measurement, use a non-zero LLM temperature where appropriate:

```bash
export PHIDO_TEMPERATURE=0.7
phido eval interpreter --trials 5
```

Evaluation results depend on the selected model, model configuration, test cases, and number of trials. They should not be interpreted as a general benchmark of the original PhIDO framework.

### 5. Use the stages from Python

The stages can also be called directly from Python. The following examples assume that the required objects (`llm`, `library`, and `pdk`) have already been configured:

```python
from phido.core.schemas import (
    InterpreterInput,
    DesignerInput,
)
from phido.stages.interpreter import InterpreterStage
from phido.stages.designer.stage import DesignerStage

# Run the Interpreter
interpreter = InterpreterStage(llm, library)
interpreter_output = interpreter.run(
    InterpreterInput(
        prompt="Two MZIs in series"
    )
)

# Run the Designer independently using the Interpreter output
designer = DesignerStage(pdk)
designer_output = designer.run(
    DesignerInput(
        template=interpreter_output
    )
)
```

The exact object construction depends on the selected LLM backend and the PDK/template data loaded by the application.

---

## Design Principles

The implementation follows several architectural principles:

1. **Typed stage boundaries:** Pydantic models are used to define the data exchanged between stages.
2. **Separation of LLM reasoning and deterministic logic:** The LLM extracts or proposes structured information, while graph construction, motif expansion, and validation are handled programmatically.
3. **Retrieval-first design:** Existing template and PDK knowledge is used before asking an LLM to resolve ambiguous component choices.
4. **Independent stages:** The Interpreter and Designer can be executed and evaluated separately.
5. **Replaceable components:** Retrieval and wiring strategies can be extended without redesigning the complete pipeline.
6. **Testability:** The mock LLM backend and offline tests support development without live model calls.

---

## Extending the Project

The architecture is designed to support further development.

### Add a circuit motif

Define a new motif function using the `@motif(...)` decorator in:

```text
src/phido/knowledge/motifs.py
```

Motif expansion is handled programmatically.

### Replace or extend retrieval

Implement the `Retriever.search(query, k, focus)` interface and provide the implementation to the relevant `PDK` or `TemplateLibrary` instance.

This makes it possible to experiment with alternative retrieval approaches, including vector-based retrieval.

### Add a wiring strategy

Implement the `EditProposer.propose(graph, pending, feedback)` interface to introduce another connection-proposal strategy.

### Add real PDK data

The included generic PDK is illustrative. A foundry-specific PDK can be introduced by supplying compatible component and port information and adapting the loader if the source data uses a different format.

---

## Current Limitations

This repository is a research-oriented reimplementation and should be understood within the scope of the current implementation.

- **No complete physical layout flow:** GDSII generation and foundry-ready physical implementation are not established by the current release.
- **No complete physical verification flow:** The repository does not claim full design-rule checking or foundry sign-off.
- **Graph-level planarity:** The planarity check is graph-level and produces a warning. Port-order heuristics are intended to avoid crossings only for trees and chains.
- **Interpreter evaluation scope:** The current Interpreter evaluation compares component kinds and connectivity; it does not yet evaluate ports or parameter values.
- **Illustrative PDK data:** Values in the generic PDK are illustrative and must not be treated as foundry-qualified component data.
- **LLM-dependent behavior:** Interpreter results can vary with the selected model, endpoint, prompt behavior, and generation settings.
- **Model validation:** The project notes identify that the implementation has not yet been tested against a real small LLM; prompt and few-shot tuning may be necessary for a particular model.
- **Not a reproduction of the original benchmark:** The implementation does not claim to reproduce the original paper's complete end-to-end benchmark results.

---

## Future Work

Potential directions for extending the project include:

- Adding more circuit motifs and reusable photonic design templates.
- Integrating additional PDKs and more comprehensive component models.
- Exploring vector-based retrieval alongside BM25.
- Improving port assignment and routing strategies for more general circuit topologies.
- Extending evaluation to include port assignments and component parameters.
- Adding physical layout generation and verification stages.
- Evaluating performance across different LLMs and generation configurations.

These are possible extensions, not features claimed to be complete in this release.

---

## Citation

If you use this repository in academic work, please cite the original research paper that inspired the project and clearly identify this repository as an independent reimplementation.

```bibtex
@article{sharma2025aiagents,
  author  = {Sharma, A. and Fu, Y. and Ansari, V. and others},
  title   = {AI agents for photonic integrated circuit design automation},
  journal = {APL Machine Learning},
  volume  = {3},
  pages   = {046113},
  year    = {2025},
  doi     = {10.1063/5.0300741}
}
```

---


## Authors & Contributors

- **[َAbbas Mohammadi Kian]** — Co-developer
- **[Co-Author: Reza Hosseini]** — Co-developer


## License

MIT — see [LICENSE](LICENSE).


