# PhIDO-Reimplementation

**An independent, from-scratch Python reimplementation of an LLM-assisted photonic integrated circuit (PIC) design framework.**

This repository rebuilds key components of **PhIDO (Photonics Intelligent Design and Optimization)** using typed, modular, independently testable stages. It is inspired by the research paper *AI agents for photonic integrated circuit design automation* (Sharma et al., 2025), but is **not** the original authors' implementation.

## Overview

The intended design workflow converts natural-language photonic circuit requirements into structured representations:

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
       Designer              (not included in this release)
          |
          v
     PIC DSL (YAML)          (planned downstream output)
```

**Current scope:** The repository contains the foundation, domain knowledge utilities, retrieval, and the Interpreter stage. The Designer, integrated pipeline, evaluation/CLI, layout generation, and verification are not part of the current release.

## Implemented Components

### Foundation

- **Typed contracts:** Pydantic schemas for structured stage inputs and outputs (`core/schemas.py`).
- **Stage interface:** Shared abstraction for modular processing stages (`core/stage.py`).
- **Run storage and errors:** Supporting utilities in `core/store.py` and `core/errors.py`.
- **LLM integration:** Structured LLM adapter, mock backend, and OpenAI-compatible client adapter in `llm/`.

### Knowledge Base & Retrieval

- **PDK knowledge:** Utilities for accessing photonic component information (`knowledge/pdk.py`).
- **Template library:** Curated circuit templates (`knowledge/templates.py` and `data/templates/templates.yaml`).
- **Motifs:** Reusable circuit patterns and programmatic motif expansion (`knowledge/motifs.py`).
- **BM25 retrieval:** Search over the template library, with a `Retriever` protocol for alternative search implementations (`knowledge/retriever.py`).
- **Text utilities:** Supporting text processing in `knowledge/text.py`.

### Interpreter

`InterpreterStage` (`stages/interpreter.py`) processes natural-language PIC requests and produces a structured `PICTemplate`.

The LLM is used to **extract entities**, such as circuit blocks, connections, or a motif name. Graph construction and motif expansion are performed in code rather than delegated entirely to the LLM. The interpreter can retrieve relevant templates using BM25 over the curated library.

## Repository Structure

```text
src/
  phido/
    core/
      schemas.py
      stage.py
      store.py
      errors.py
    llm/
      adapter.py
      mock.py
      openai_compat.py
    knowledge/
      pdk.py
      templates.py
      motifs.py
      retriever.py
      text.py
    stages/
      interpreter.py
data/
  pdk/
    generic_pdk.yaml
  templates/
    templates.yaml
tests/
pyproject.toml
LICENSE
README.md
```

## Installation and Tests

From the repository root, install the project with development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The included tests are designed to run without network access or live LLM calls.

## Interpreter Usage

The following example shows the Interpreter interface. It assumes `llm` has already been configured as a `StructuredLLM` instance (see `phido/llm/adapter.py` and `phido/llm/openai_compat.py`).

```python
from phido.core.schemas import InterpreterInput
from phido.knowledge.templates import TemplateLibrary
from phido.stages.interpreter import InterpreterStage

library = TemplateLibrary.load("data/templates/templates.yaml")
out = InterpreterStage(llm, library).run(
    InterpreterInput(prompt="Two MZIs in series")
)
```

## Extending the Implementation

- **Add a circuit motif:** Define a function with `@motif(...)` in `knowledge/motifs.py`.
- **Replace retrieval:** Implement `Retriever.search(query, k, focus)` and supply the retriever to `PDK(...)` or `TemplateLibrary(...)`.
- **Add processing stages:** Extend the stage-based architecture while retaining typed contracts and independent tests.

## Current Limitations

This release does **not** establish complete natural-language-to-GDSII generation or reproduce the paper's full end-to-end benchmark results. Those claims require the downstream design, layout, verification, and evaluation components and their corresponding results to be published and documented.

## Authors & Contributors

- **[َAbbas Mohammadi Kian]** — Co-developer
- **[Co-Author: Reza Hosseini]** — Co-developer

## Original Research & Attribution

Sharma, A., Fu, Y., Ansari, V., *et al.* (2025). **AI agents for photonic integrated circuit design automation.** *APL Machine Learning*, **3**, 046113.

- Paper: https://doi.org/10.1063/5.0300741
- Original authors' repository: https://github.com/JPPhotonics/PhIDORelease

This is an **independent research reimplementation**, not an official PhIDO release or an affiliated project.

## License

MIT — see [LICENSE](LICENSE).


