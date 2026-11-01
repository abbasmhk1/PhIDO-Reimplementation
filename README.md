# PhIDO-Reimplementation
**Independent reimplementation of an AI-assisted photonic integrated circuit (PIC) design framework.**

This repository documents our independent, from-scratch Python reimplementation inspired by **PhIDO (Photonics Intelligent Design and Optimization)**, described in *AI agents for photonic integrated circuit design automation* (Sharma et al., 2025).

This repository contains the core software foundation for the implementation, including typed interfaces, run storage, and LLM integration. The features described below reflect the scope of the code documented here.

## Implemented components

- **Typed schemas:** Pydantic data models for information exchanged between pipeline stages (`core/schemas.py`).
- **Stage abstraction:** a reusable base interface for independently testable processing stages (`core/stage.py`).
- **Run storage:** a `RunStore` component for managing intermediate run data (`core/store.py`).
- **Error definitions:** shared exceptions for pipeline components (`core/errors.py`).
- **LLM adapter:** a common interface for generation, output validation, and retries (`llm/adapter.py`).
- **Model backends:** a mock implementation for offline tests and an OpenAI-compatible adapter (`llm/mock.py`, `llm/openai_compat.py`).
- **Tests and CI:** unit tests for the foundation and a GitHub Actions workflow.

### Project structure

```text
.
├── src/
│   └── phido/
│       ├── core/
│       │   ├── errors.py
│       │   ├── schemas.py
│       │   ├── stage.py
│       │   └── store.py
│       └── llm/
│           ├── adapter.py
│           ├── mock.py
│           └── openai_compat.py
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   └── test_llm_adapter.py
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Installation and testing

Requires **Python 3.10+**.

```bash
# From the repository root
python -m pip install -e ".[dev]"
python -m pytest
```

The included tests use mock components and do not require live LLM API calls.

## Design principles

1. **Modularity:** each stage has an explicit interface and can be tested independently.
2. **Structured outputs:** Pydantic models define the contracts between stages.
3. **Reliability:** model-generated responses are validated through a shared adapter with retry handling.
4. **Reproducibility:** local tests can run without network access or paid model calls.

## Current scope and limitations

The components documented here establish the software infrastructure. This version **does not yet demonstrate complete natural-language-to-GDSII generation**, circuit simulation, or reproduction of the paper's benchmark results.

## Authors & Contributors

- **[Abbas Mohammadikian]** — Co-developer
- **[Co-Author:Reza hosseini]** — Co-developer

Contribution details can be expanded as the repository develops.

## Original research and attribution

This project is based on:

Sharma, A., Fu, Y., Ansari, V., *et al.* (2025). **AI agents for photonic integrated circuit design automation.** *APL Machine Learning*, **3**, 046113. https://doi.org/10.1063/5.0300741

Original authors' repository: https://github.com/JPPhotonics/PhIDORelease

**Disclaimer:** This is an independent educational/research implementation and is not the official PhIDO release or affiliated with the original authors.

## License

MIT — see [LICENSE](LICENSE).

