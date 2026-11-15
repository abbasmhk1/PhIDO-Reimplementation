# PhotonWeave
### Hybrid, Physics-Aware Photonic Circuit Design Automation

**From natural-language specifications to circuit graphs, PDK-based designs, GDSII layouts, and optical simulation.**

PhotonWeave is an open, modular framework for automating photonic integrated circuit (PIC) design. It translates natural-language descriptions into structured circuit graphs, selects components from a photonic process design kit (PDK), constructs physically meaningful connections, generates layouts, and evaluates circuit behavior through optical simulation.

Its central design principle is simple: **use language models where semantic understanding is needed, and deterministic engineering algorithms where explicit rules, geometry, and physical constraints govern the task.**

Rather than delegating the entire design process to an LLM, PhotonWeave combines language models with symbolic graph construction, pattern expansion, information retrieval, geometry-aware connectivity, automated layout validation, and an independently implemented S-matrix circuit solver.

The framework is designed to make photonic design automation modular, reproducible, extensible, and independently evaluable at every stage.

---

## Table of Contents


- [PhotonWeave](#photonweave)
    - [Hybrid, Physics-Aware Photonic Circuit Design Automation](#hybrid-physics-aware-photonic-circuit-design-automation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [From description to photonic circuit](#from-description-to-photonic-circuit)
  - [Why PhotonWeave?](#why-photonweave)
    - [1. Constrained LLM-based interpretation](#1-constrained-llm-based-interpretation)
    - [2. Retrieval-assisted PDK component selection](#2-retrieval-assisted-pdk-component-selection)
    - [3. Geometry- and physics-aware connectivity](#3-geometry--and-physics-aware-connectivity)
    - [4. Automated layout with structural validation](#4-automated-layout-with-structural-validation)
    - [5. Independent circuit simulation](#5-independent-circuit-simulation)
    - [6. Stage-isolated evaluation and reproducibility](#6-stage-isolated-evaluation-and-reproducibility)
  - [Architecture](#architecture)
    - [Interpreter: Natural language to circuit graph](#interpreter-natural-language-to-circuit-graph)
    - [Designer: Circuit graph to configured design](#designer-circuit-graph-to-configured-design)
    - [Layout: Configured design to GDSII](#layout-configured-design-to-gdsii)
    - [Verify: Circuit design to optical response](#verify-circuit-design-to-optical-response)
  - [Quick Start](#quick-start)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Run without an LLM](#run-without-an-llm)
    - [Run with an LLM](#run-with-an-llm)
  - [Example](#example)
    - [Generated circuit](#generated-circuit)
    - [Example output](#example-output)
    - [More examples](#more-examples)
    - [Plotting a spectrum](#plotting-a-spectrum)
    - [Writing effective specifications](#writing-effective-specifications)
  - [Design Principles](#design-principles)
    - [Typed circuit intermediate representation](#typed-circuit-intermediate-representation)
  - [Evaluation Framework](#evaluation-framework)
    - [Metrics](#metrics)
    - [Current benchmark suites](#current-benchmark-suites)
  - [Results So Far](#results-so-far)
    - [Software and stage results](#software-and-stage-results)
    - [Numerical validation](#numerical-validation)
    - [Engineering lessons from development](#engineering-lessons-from-development)
  - [Configuration](#configuration)
    - [Model requirements](#model-requirements)
  - [Commands](#commands)
    - [Individual stages](#individual-stages)
    - [Full pipeline](#full-pipeline)
    - [Run a built-in template](#run-a-built-in-template)
    - [List available templates](#list-available-templates)
    - [Evaluate stages](#evaluate-stages)
    - [Compare models](#compare-models)
    - [Reproducible stage execution](#reproducible-stage-execution)
  - [Project Structure](#project-structure)
  - [Extending PhotonWeave](#extending-photonweave)
    - [Adding a circuit pattern](#adding-a-circuit-pattern)
    - [Adding photonic components](#adding-photonic-components)
    - [Adding a retrieval strategy](#adding-a-retrieval-strategy)
    - [Adding a wiring strategy](#adding-a-wiring-strategy)
    - [Using custom data](#using-custom-data)
    - [Adding a processing stage](#adding-a-processing-stage)
  - [Limitations](#limitations)
    - [Demo PDK and component models](#demo-pdk-and-component-models)
    - [Structural layout checks, not foundry DRC](#structural-layout-checks-not-foundry-drc)
    - [Simplified device geometry](#simplified-device-geometry)
    - [Benchmark limitations](#benchmark-limitations)
    - [No direct reproduction of the reference benchmark](#no-direct-reproduction-of-the-reference-benchmark)
    - [Features not yet implemented](#features-not-yet-implemented)
  - [Troubleshooting](#troubleshooting)
  - [Relationship to Previous Work](#relationship-to-previous-work)
  - [Authors](#authors)
  
---

## Overview

A photonic integrated circuit involves more than selecting optical components and connecting their ports. A usable design must account for circuit topology, component characteristics, port compatibility, optical behavior, physical geometry, routing constraints, and circuit-level performance.

PhotonWeave addresses these requirements through a modular pipeline:

1. **Interpretation:** Extract circuit entities, connections, parameters, and repeating structures from natural-language specifications.
2. **Circuit construction:** Expand symbolic patterns into explicit graphs and identify suitable components from a PDK.
3. **Connectivity:** Configure parameters and construct geometry-aware, physics-informed port connections.
4. **Layout:** Place components, route waveguides, validate structural constraints, and generate GDSII.
5. **Verification:** Simulate optical transmission and inspect circuit-level behavior.

### From description to photonic circuit

```text
Natural-language specification
             |
             v
      INTERPRETER
  LLM-based semantic extraction
  Symbolic pattern recognition
             |
             v
       CIRCUIT GRAPH
  Explicit components and connections
  Hierarchy and pattern expansion
             |
             v
        DESIGNER
  PDK retrieval and component selection
  Parameter configuration
  Geometry/physics-aware connectivity
             |
             +--------------------+
             |                    |
             v                    v
          LAYOUT               VERIFY
     Component placement     S-matrix solver
     Waveguide routing       Optical simulation
     Structural checks       Circuit diagnostics
             |                    |
             v                    v
           GDSII              Spectra and
          OUTPUT              diagnostics
```

Each stage operates through validated, typed data contracts. Its inputs and outputs are stored as reproducible artifacts, allowing individual stages to be executed, inspected, evaluated, and rerun independently.

---

## Why PhotonWeave?

PhotonWeave is built around a hybrid engineering architecture rather than unrestricted LLM-driven design.

Its distinguishing capabilities include:

### 1. Constrained LLM-based interpretation

Language models extract design intent rather than constructing the entire circuit themselves.

The interpreter identifies:

- Component types and instances.
- Connections and relationships.
- Component and circuit parameters.
- Repeating structures such as trees, cascades, and interferometer meshes.
- Optional input and output couplers.

Deterministic code constructs the corresponding graph and expands repeating patterns. This reduces reliance on LLM-generated component enumeration and hierarchical connectivity.

### 2. Retrieval-assisted PDK component selection

Component selection combines information retrieval with selective language-model reasoning.

- BM25 retrieval ranks candidate components using PDK documentation and descriptions.
- Language models resolve ambiguous or closely ranked candidates.
- Parameter aliases are mapped to canonical names.
- Parameter ranges and validity are checked deterministically.

This architecture avoids requiring the language model to reason over the entire PDK for every selection.

### 3. Geometry- and physics-aware connectivity

Port connectivity is not treated as a purely syntactic operation.

PhotonWeave considers:

- Component port definitions.
- Port ordering and orientation.
- Placement geometry.
- Crossing avoidance.
- The modeled optical behavior of components.

For components in an optical chain, connectivity follows the path carrying light in the component's modeled default state.

Deterministic wiring algorithms construct and validate connections, while language-model assistance is available for ambiguous decisions.

### 4. Automated layout with structural validation

The layout engine combines layered placement, port alignment, routing, and structural checks.

It identifies issues such as:

- Component overlaps.
- Routing failures.
- Wire crossings.
- Wires passing through other components.
- Overlapping sibling routes.

When routing defects are detected, the engine can retry with increased spacing.

### 5. Independent circuit simulation

PhotonWeave includes its own NumPy-based S-matrix circuit solver and analytical component models.

The verifier evaluates transmission across wavelength sweeps and reports:

- Optical power distribution.
- Unconnected optical ports.
- Dark outputs.
- Non-passive circuit behavior.
- Transmission spectra between external ports.

Its connection logic has also been cross-checked against SAX for the tested configurations.

### 6. Stage-isolated evaluation and reproducibility

PhotonWeave includes an evaluation framework that measures individual stages using gold inputs.

This makes it possible to distinguish interpreter failures from component-selection, parameter-configuration, schematic-generation, layout, and verification failures.

The evaluation framework also tracks model calls, token usage, execution time, and failure categories.

---

## Architecture

PhotonWeave consists of four independently executable stages, supported by shared infrastructure.

| Stage | Main responsibilities | LLM dependency |
|---|---|---|
| Interpreter | Semantic extraction, template retrieval, pattern identification | Required |
| Designer | PDK retrieval, component selection, parameters, connectivity | Optional/selective |
| Layout | Placement, routing, structural validation, GDSII | Not required |
| Verify | S-matrix simulation and circuit diagnostics | Not required |

### Interpreter: Natural language to circuit graph

The interpreter separates semantic understanding from graph construction.

The LLM extracts entities such as:

- Blocks and components.
- Explicit connections.
- Named repeating patterns.
- Circuit-level parameters.

Supported patterns include:

- Trees.
- Cascades.
- Clements meshes.
- Reck meshes.

Pattern generators expand these descriptions into explicit circuit graphs.

A BM25 retriever suggests similar circuit templates from the built-in library.

All LLM responses pass through a common adapter that validates structured output and retries with validation feedback when necessary. This adapter is model-independent and works with supported OpenAI-compatible endpoints.

The resulting circuit representation is a typed, validated intermediate structure rather than free-form LLM-generated code.

### Designer: Circuit graph to configured design

The designer transforms the circuit graph into a configured photonic circuit.

Its responsibilities include:

- PDK component discovery.
- Component selection.
- Parameter mapping and validation.
- Port-level connectivity.
- Circuit consistency checks.

BM25 retrieval ranks candidate PDK cells using their documentation and descriptions. An LLM is consulted when candidate selection is ambiguous.

Parameter aliases map user-facing names to canonical PDK parameters, and configured values are checked against declared ranges.

Connectivity is constructed through validated graph edits. Geometry and component behavior inform port assignments and optical path selection.

Ambiguous decisions can be reported as warnings instead of silently producing an uncertain design.

### Layout: Configured design to GDSII

The layout engine generates a physical representation of the configured circuit.

Its main operations include:

- Layered component placement.
- Port alignment.
- Stand-in tracks for long connections.
- Staggered waveguide stubs.
- Waveguide routing through gdsfactory.
- Structural overlap and crossing checks.

The router can retry with increased spacing when structural defects are detected.

The current implementation uses its own layered placement logic and gdsfactory routing.

### Verify: Circuit design to optical response

The verification stage uses an independently implemented S-matrix solver with analytical component models.

It calculates circuit transmission over a wavelength sweep and evaluates the distribution of optical power between external ports.

It also reports:

- Light leaving through unconnected ports.
- Outputs receiving no optical power.
- Non-passive circuit behavior.

The solver's connection logic has been numerically compared with SAX, with agreement to approximately machine precision in the tested configurations.

This numerical agreement validates the tested solver implementations against each other; it does not establish foundry-level physical accuracy for the illustrative device models.

---

## Quick Start

### Requirements

- Python 3.10 or newer.
- An LLM endpoint for natural-language interpretation.
- gdsfactory for GDSII layout generation.
- pytest for development and testing.

### Installation

```bash
pip install -e ".[dev,layout]"
```

Run the test suite:

```bash
pytest -q
```

The tests run offline and do not require an LLM.

### Run without an LLM

You can use an existing circuit template and execute the downstream design stages without language-model inference.

```bash
phido templates
phido pipeline --adopt edge_mzi_edge --layout --verify
```

### Run with an LLM

PhotonWeave supports OpenAI-compatible endpoints, including locally hosted Ollama models.

```bash
export PHIDO_LLM_MODEL=qwen3:8b-8k

phido pipeline \
  "An edge coupler feeds a 2x2 MZI, whose output goes to a photodetector." \
  --layout \
  --verify
```

The default endpoint is:

```text
http://localhost:11434/v1
```

Results are stored under:

```text
runs/<id>/
```

Typical outputs include:

```text
runs/<id>/
    interpreter/
    designer/
    layout/
    verify/
```

Each stage stores its input, output, and execution information.

---

## Example

Consider the following request:

> A 1x4 power splitter made of 1x2 MMIs, with an edge coupler at the input and a photodetector at every output.

PhotonWeave interprets the repeating structure and constructs the corresponding graph.

### Generated circuit

```text
                 MMI_1
                /     \
               /       \
          MMI_2         MMI_3
          /   \         /   \
        PD1   PD2     PD3   PD4

          ^
          |
      Edge coupler
```

### Example output

```text
doc:
  title: 2-ary tree of mmi, 2 stages

nodes:
  N1: mmi_1x2
  N2: mmi_1x2
  N3: mmi_1x2
  IN1: edge_coupler
  OUT1..OUT4: photodetector

layout:
  components: 8
  routes: 7/7 successful
  output: GDSII

verify:
  inputs: [IN1,fiber]
  lit outputs: 4
  unconnected ports: 0
```

Illustrative transmission results:

```text
IN1,fiber -> OUT1,pd:
  -8.12 dB at centre
  peak: -8.12 dB
  wavelength: 1.5 um

IN1,fiber -> OUT2,pd:
  -8.12 dB

IN1,fiber -> OUT3,pd:
  -8.12 dB

IN1,fiber -> OUT4,pd:
  -8.12 dB
```

The expected power budget is approximately:

- Ideal 1:4 splitting: 6.02 dB.
- Two MMI excess losses: 0.6 dB.
- Input edge coupler loss: 1.5 dB.

Total: approximately 8.12 dB.

These values illustrate the configured demo component models, not measured device performance.

### More examples

| Natural-language specification | Expected behavior |
|---|---|
| An edge coupler feeds a 2x2 MZI, whose output goes to a photodetector. | Transmission approximately -1.52 dB with no expected structural warnings |
| An edge coupler feeds a 2x2 MZI with a path length difference of 100 um. | Interference fringes with approximately 5.72 nm spacing and a warning for the unconnected complementary output |
| An edge coupler feeds an add-drop ring resonator; connect the through port to a photodetector. | Resonance notches with approximately 9.1 nm spacing and a warning for the unconnected drop port |
| A 4-mode Clements mesh of MZIs with an edge coupler at every input and output. | 14 components and 16/16 successful routes |

The examples use the current illustrative PDK and analytical models.

### Plotting a spectrum

For narrow spectral features such as ring resonances, use a sufficiently dense wavelength sweep.

```bash
phido --no-store verify \
  runs/<id>/designer/output.yaml \
  --points 2001 \
  --csv spectrum.csv
```

### Writing effective specifications

For reliable interpretation:

- Name the components explicitly: MZI, MMI, ring resonator, edge coupler, grating coupler, photodetector, heater, and directional coupler.
- Specify port counts, such as 1x2 or 2x2.
- Include numerical values with units.
- Specify coupler placement, such as input only or every output.
- Identify the intended MZI technology when relevant, such as thermo-optic or PIN-diode.
- Name repeating structures explicitly when applicable.

Supported repeating structures include trees, cascades, and Clements/Reck meshes.

---

## Design Principles

PhotonWeave follows several architectural principles.

| Principle | Implementation |
|---|---|
| Structured before generated | Natural-language descriptions become explicit, validated circuit graphs |
| Deterministic where possible | Pattern expansion, parameter checks, connectivity and geometry use algorithms |
| LLMs where useful | Language models handle semantic extraction and ambiguous decisions |
| Physics-aware construction | Component behavior informs optical path selection and connectivity |
| Independently verifiable stages | Every stage has typed inputs, outputs and separate execution |
| Reproducible execution | Intermediate artifacts, LLM traces and errors are stored |
| Extensible engineering | PDKs, templates, motifs, retrievers and component models are modular |

### Typed circuit intermediate representation

The intermediate representation is central to the architecture.

It provides a structured interface between natural-language interpretation and downstream engineering operations.

It captures:

- Component instances.
- Ports and connections.
- Circuit topology.
- Parameters.
- Repeating structures and expanded graphs.

Each processing stage uses Pydantic-defined contracts to validate its input and output.

This separation allows stages to be tested, replaced, and developed independently without requiring changes to the entire pipeline.

---

## Evaluation Framework

PhotonWeave includes a stage-isolated evaluation harness.

Instead of relying exclusively on end-to-end success, each stage can be evaluated independently using gold inputs. This prevents upstream failures from obscuring the capabilities of downstream stages.

The evaluation framework follows the outcome-code structure used in the reference research.

| Code | Evaluation target |
|---|---|
| EE | Entity extraction |
| CS | Component selection |
| PC | Parameter configuration |
| SG | Schematic generation |
| L | Layout |
| V | Verification |

### Metrics

The framework reports:

- Pass@k.
- Absolute occurrence rates.
- Conditional occurrence rates.
- LLM calls.
- Token consumption.
- Execution time.
- Stage-specific errors.

A model-comparison command allows different language models to be evaluated under the same benchmark configuration.

### Current benchmark suites

| Suite | Cases | Purpose |
|---|---:|---|
| Basic | 13 | Single components, short chains and simple trees |
| Hard | 18 | Hierarchy, enumeration, per-node parameters, patterns and meshes |
| All | 31 | Combined development suite |
| Holdout | 16 | Separate evaluation suite, excluded from prompt tuning |

The current benchmark contains 47 interpreter prompts written for this project. It is not the original PhIDO benchmark.

The holdout suite was written after tuning the development suite and is kept separate from it.

The scorer checks components, connectivity, and any ports or parameters specified by the gold case. Parameter aliases are handled through the PDK alias table.

These suites are currently intended for internal development and comparative evaluation. They are not yet an independently authored benchmark.

---

## Results So Far

The following results are from the current implementation and local Ollama inference environment.

They are preliminary engineering results, not a controlled comparison with the original PhIDO paper.

### Software and stage results

| Evaluation | Result |
|---|---|
| Unit and integration tests | 136 passing, offline |
| Interpreter, basic + hard, qwen3.8:27b-8k | 62/62 across two trials per case |
| Interpreter, held-out suite, qwen3.8:27b-8k | 16/16 |
| Interpreter, held-out suite, qwen3-coder:30b | 16/16 |
| Designer, LLM-assisted configuration, 9 gold cases | 9/9 across two trials per case |
| Clements mesh layout, 2–8 modes | Clean structural checks, up to 44 components |

### Numerical validation

| Test | Result |
|---|---|
| MZI fringe spacing | Approximately 5.72 nm, consistent with theory |
| Ring resonance spacing | Approximately 9.14 nm, compared with approximately 9.10 nm theoretical |
| S-matrix connection logic | Agreement with SAX to approximately 1e-16 in tested configurations |

### Engineering lessons from development

Systematic evaluation and regression testing identified several defects that were subsequently corrected:

- Missing support for input-only and output-only coupler placement.
- Null-handling problems.
- Hierarchy and enumeration errors.
- Incorrect wiring to unused optical ports.
- Routing constraints involving vertical jog lengths.
- Overlapping sibling routes.
- Routes passing through other components.

These results demonstrate the usefulness of systematic stage-level testing, but the benchmark remains small.

The current results do not establish statistically meaningful performance differences between strong models. The held-out suite also remains too small to establish broad generalization.

---

## Configuration

The LLM is required for natural-language interpretation. The downstream stages can run without an LLM.

| Variable | Description | Default |
|---|---|---|
| `PHIDO_LLM_MODEL` | Language model identifier | Required for LLM operations |
| `PHIDO_LLM_BASE_URL` | OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `PHIDO_LLM_API_KEY` | API key when required | None |
| `PHIDO_TEMPERATURE` | Sampling temperature | `0` |
| `PHIDO_NO_THINK` | Disable Qwen3 thinking mode through `/no_think` | Off |
| `PHIDO_DATA` | Alternative PDK, template and test-data directory | `./data` |

Global CLI options:

```bash
--runs-dir DIR
--no-store
```

### Model requirements

The interpreter prompt and schema require approximately 2–3k tokens of context. Use a model with at least an 8k context window.

The implementation supports different OpenAI-compatible endpoints, including local and remote Ollama servers.

For remote inference, an SSH tunnel can be used:

```bash
ssh -N -L 11435:HOST:11434 user@gateway
```

Then configure:

```bash
export PHIDO_LLM_BASE_URL=http://localhost:11435/v1
```

Reasoning-model `<think>` blocks are stripped before structured response parsing.

---

## Commands

### Individual stages

```bash
# Natural language to circuit template
phido interpret "PROMPT"

# Template to configured photonic design
phido design template.yaml

# Optional LLM assistance for design decisions
phido design template.yaml --llm

# Configured design to layout and GDSII
phido layout designer_output.yaml

# Circuit simulation
phido verify designer_output.yaml --csv spectrum.csv
```

### Full pipeline

```bash
phido pipeline "PROMPT" --layout --verify
```

### Run a built-in template

```bash
phido pipeline --adopt edge_mzi_edge --layout --verify
```

### List available templates

```bash
phido templates
```

### Evaluate stages

```bash
phido eval designer
phido eval layout
phido eval verify
phido eval interpreter --suite holdout --trials 2
```

### Compare models

```bash
phido compare --models model_a,model_b --suite all --trials 2
```

### Reproducible stage execution

Each run stores its intermediate artifacts:

```text
runs/<id>/
    interpreter/
        input.yaml
        output.yaml
        llm_trace.json

    designer/
        input.yaml
        output.yaml
        llm_trace.json

    layout/
        input.yaml
        output.yaml

    verify/
        input.yaml
        output.yaml
```

When a stage fails, an `error.txt` file records the failure.

Individual stages can be rerun from saved artifacts. Intermediate YAML files can also be inspected or edited before downstream execution.

---

## Project Structure

```text
src/phido/
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

    geometry/
        placement.py

    stages/
        interpreter.py

        designer/
            selector.py
            params.py
            circuit.py
            proposers.py
            crossings.py
            throughput.py
            stage.py

        layout/
            stage.py
            gds.py

        verify/
            solver.py
            models.py
            stage.py

    eval/
        metrics.py
        runner.py
        compare.py

    pipeline.py
    cli.py

data/
    pdk/
        generic_pdk.yaml

    templates/
        templates.yaml

    testbench/

tests/
```

---

## Extending PhotonWeave

The architecture is designed to support new components, PDKs, patterns, retrieval strategies and processing stages.

### Adding a circuit pattern

Implement a motif function in:

```text
knowledge/motifs.py
```

Register it using the motif interface. Registered patterns are made available to the interpreter.

### Adding photonic components

Add component definitions to:

```text
data/pdk/generic_pdk.yaml
```

A component definition includes:

- Port descriptions.
- Parameter definitions.
- Valid parameter ranges.
- Parameter aliases.
- Simulation model identifiers.
- Layout configuration.

Add the corresponding analytical model to:

```text
stages/verify/models.py
```

### Adding a retrieval strategy

Implement the retriever interface:

```python
Retriever.search(query, k, focus)
```

This makes it possible to experiment with alternative approaches, including vector-based retrieval for larger PDKs.

### Adding a wiring strategy

Implement:

```python
EditProposer.propose(graph, pending, feedback)
```

The strategy can then be integrated into the designer's connectivity workflow.

### Using custom data

Set:

```bash
export PHIDO_DATA=/path/to/your/data
```

The alternative directory must follow the expected PDK, template and testbench YAML structures.

### Adding a processing stage

Implement a new stage using the shared `Stage` interface and define its Pydantic input and output contracts.

Integrate it into the pipeline as required.

---

## Limitations

PhotonWeave is a research and engineering prototype. Its current implementation has several important limitations.

### Demo PDK and component models

The included 18-cell PDK is illustrative.

Current component models use simplified constants, including:

- Effective refractive index: `n_eff = 2.4`
- Group index: `n_g = 4.2`
- Propagation loss: `2 dB/cm`
- Illustrative ring coupling parameters.

These values are not foundry-qualified device data. Absolute performance predictions should not be treated as measurements or fabrication-ready results.

### Structural layout checks, not foundry DRC

The layout engine performs structural checks for overlaps, crossings, routing failures and component interference.

It does not currently implement a foundry-specific design-rule-checking deck.

Reck meshes with four or more modes currently produce one flagged route overlap per tested layout.

The issue is reported rather than silently ignored.

### Simplified device geometry

Some component models are currently simplified:

- `y_branch` is represented as a black box.
- PIN and PN MZIs reuse passive MZI geometry.

### Benchmark limitations

The current benchmark contains 47 interpreter prompts written by the project authors.

It is not the original PhIDO benchmark and has not yet been independently authored or externally validated.

The current results are preliminary and should not be interpreted as a statistically established performance advantage over other systems.

### No direct reproduction of the reference benchmark

The implementation has not been evaluated on the original PhIDO testbench, PDK or templates.

Therefore, the reported numbers are not directly comparable with the results in the reference publication.

### Features not yet implemented

The following capabilities remain future work:

- Large-scale PDK evaluation and retrieval studies.
- Vector-based PDK retrieval.
- Systematic mixed-model evaluation across stages.
- Foundry-specific DRC integration.
- Performance-qualified circuit generation through iterative optimization.
- Experimental or measured-device validation.

---

## Troubleshooting

| Problem | Possible cause and solution |
|---|---|
| `Set PHIDO_LLM_MODEL` | Set the model environment variable in the current shell |
| LLM endpoint returns HTTP 404 | Check the endpoint URL and whether the requested model exists |
| Empty or truncated model responses | Increase the context window or use a model with at least 8k context |
| Reasoning model returns `<think>` text | Configure `PHIDO_NO_THINK=1` where supported |
| `gdsfactory is not installed` | Install the layout dependencies using `pip install -e ".[layout]"` |
| Multiple equally ranked components | Specify the component technology or port configuration more explicitly |
| Light leaves through an unconnected port | Inspect the component's other optical ports and connect them if required |
| Ring resonance notches appear shallow | Increase spectral sampling density or narrow the wavelength range |
| `phido: command not found` | Activate the environment where the package was installed |

For ring-resonator simulations, use a dense wavelength sweep:

```bash
--points 2001
```

For an Ollama endpoint, check available models using:

```bash
ollama list
```

---

## Relationship to Previous Work

PhotonWeave is motivated by the research direction of natural-language-driven photonic circuit design automation introduced in:

Sharma, A., Fu, Y., Ansari, V., Iyer, R., Kuang, F., Mistry, K., Aishy, R. I., Ahmad, S., Matres, J., Englund, D. R., and Poon, J. K. S.

**AI agents for photonic integrated circuit design automation.**

APL Machine Learning, 3, 046113 (2025).

https://doi.org/10.1063/5.0300741

The two projects address a related automation problem but use different implementations and architectural choices.

PhotonWeave does not use the authors' source code, benchmark data, or original PDK.

Its architecture emphasizes:

- Typed intermediate representations.
- Deterministic symbolic graph construction.
- Retrieval-assisted component selection.
- Geometry- and physics-aware connectivity.
- Independent layered placement and structural validation.
- An independently implemented S-matrix circuit solver.
- Stage-isolated evaluation and reproducible execution.

The current implementation has not been benchmarked under the original paper's experimental conditions, so direct performance comparisons are not claimed.

The original PhIDO project and publication should be credited as prior research motivating this work.

---



## Authors

**PhotonWeave** is an independent research and software development project created by:

- **Abbas Mohammadi Kian**
- **Sayed Reza Hoseini**
  
Both authors contributed to the development of the project.



**PhotonWeave — Combining language understanding with deterministic photonic engineering.**

