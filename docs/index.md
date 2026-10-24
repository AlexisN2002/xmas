# xmas

xmas is the multi-agent sibling of
[promptolution](https://github.com/gepromptet/promptolution). It keeps the same five
nouns, Task, Predictor, LLM, Optimizer and ExperimentConfig, and lifts them one level:
the multi-agent system itself is the Predictor, and its per-agent prompts, its topology
and its per-node model bindings are all optimizable components. A single-agent system is
the exact degenerate case, so promptolution's optimizers plug in unchanged.

!!! note "Beta"

    Inference, the optimization loop, CANTANTE, topology search, code steps, JSON
    answers, conditions, loops, per-item chains, media values, budgets, health,
    people in the graph, Stop, `first of` races, the `.mas` bundle, `verify()`
    and the `xmas` command are in place. Learning from human picks and the
    experiment layer are still to come.

## Where to start

- [Design principles](design-principles.md), the six rules the design answers to.
- [API reference](api.md), generated from docstrings.
- [Contributing](https://github.com/gepromptet/xmas/blob/main/CONTRIBUTING.md).

## Configuration

A saved system names its models and tool servers and never holds an endpoint or a
key. Those come from the environment, then from `~/.config/xmas/config.toml` (or
the file `XMAS_CONFIG` names); see `xmas.config` in the API reference.

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |
| `XMAS_MODELS` | Model ids, comma separated, that stand for the catalogue |
| `XMAS_CONFIG` | The configuration file |
| `XMAS_TOOLS_KEY` | Key for a tool server whose entry has none |
| `XMAS_STORE` | Folder for stored media |
| `XMAS_RUNS` | File of the run store |

Copy `.env.example` to `.env` and fill in your own values. `.env` is gitignored.
xmas never opens `.env` itself: export the variables, or run with
`uv run --env-file .env`.
