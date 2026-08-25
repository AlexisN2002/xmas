# xmas

xmas is the multi-agent sibling of
[promptolution](https://github.com/gepromptet/promptolution). It keeps the same five
nouns, Task, Predictor, LLM, Optimizer and ExperimentConfig, and lifts them one level:
the multi-agent system itself is the Predictor, and its per-agent prompts, its topology
and its per-node model bindings are all optimizable components. A single-agent system is
the exact degenerate case, so promptolution's optimizers plug in unchanged.

!!! warning "Pre-alpha"

    The packaging shell is in place; the framework core is under construction. These
    docs will fill in as the API lands.

## Where to start

- [Design principles](design-principles.md), the six rules the design answers to.
- [API reference](api.md), generated from docstrings.
- [Contributing](https://github.com/gepromptet/xmas/blob/main/CONTRIBUTING.md).

## Configuration

xmas reads its settings from the environment and never from a file in the repository.

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |

Copy `.env.example` to `.env` and fill in your own values. `.env` is gitignored.
