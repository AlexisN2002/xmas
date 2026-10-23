# 🎄 xmas: multi-agent systems, wrapped and ready 🎁

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**xmas builds, runs and optimizes a multi-agent system as one object.** A system is a
set of agents with their prompts, the edges that wire them together and the model bound
to each node. All three are ordinary optimizable parameters, so an optimizer can improve
the prompts, the topology and the model choice against a task you care about, and every
run leaves a full per-agent trace. A single agent is the smallest system, not a special
case.

xmas is the multi-agent sibling of [promptolution](https://github.com/gepromptet/promptolution):
it keeps the same five nouns, Task, Predictor, LLM, Optimizer and ExperimentConfig, and
lifts them one level, the multi-agent system itself being the Predictor. Because xmas
builds directly on promptolution's base classes rather than reimplementing them, a
single-agent system is not an analogy to a promptolution predictor, it is exactly one,
and promptolution's optimizers plug in without a port. Within the gepromptet ecosystem
that makes promptolution the N=1 case and xmas the general one.

See [docs/design-principles.md](docs/design-principles.md) for the six principles that
hold the design together.

> **Status: pre-alpha.** Core inference and the optimization loop are in place.
> Attribution methods, topology search and the experiment layer are not.

## Installation

xmas is not published to a package index. Install from source:

```bash
git clone https://github.com/gepromptet/xmas.git
cd xmas
uv sync
```

The core depends on `promptolution` and little else. Heavy backends live behind
extras, mirroring promptolution's own `[api]` / `[vllm]` / `[transformers]` split:

```bash
uv sync --extra vllm     # vLLM backend
```

## Quickstart

```python
from xmas import Agent, MAS, APILLM

llm = APILLM(model="llama-3.1-8b", base_url="https://api.tomzehle.com/v1")

solver = Agent("solver", prompt="Solve step by step.", output="draft")
checker = Agent("checker", prompt="Verify and answer.", inputs=["query", "draft"], output="prediction")

mas = MAS([solver, checker], edges=[("solver", "checker")], llm=llm)

answer = mas("What is 17 * 23?")  # __call__ runs it
trace = mas.last_trace  # full per-agent record of that call

mas.to_yaml("gsm8k.yaml")  # serialization as methods
mas2 = MAS.from_yaml("gsm8k.yaml", llm=llm)
new = mas.copy(prompts={"solver": "..."})  # optimizers derive candidates this way
```

Agents are addressable and mutable through the system:

```python
mas.agents["solver"].prompt = "Solve carefully, showing each step."
```

## The degenerate case

A one-agent system is a promptolution predictor, not an analogue of one, so
promptolution's own optimizers run against it with no adapter:

```python
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.prompt import Prompt

mas = MAS([Agent("solver", prompt="Classify: {query}", output="prediction")], llm=llm)
task = ClassificationTask(df, x_column="x", y_column="y")

result = task.evaluate(Prompt("Classify: {query}"), predictor=mas)
```

## Optimizing

Every part of a system is a component an optimizer can change. The loop is
propose, evaluate through the task, select:

```python
from xmas.optimization import PrefixSearchOptimizer
from xmas.tasks.gsm8k import GSM8KTask

task = GSM8KTask.from_huggingface(split="test", n=50)
best = PrefixSearchOptimizer(mas, task).optimize(n_steps=3)
```

`PrefixSearchOptimizer` is deliberately trivial and exists as a control: it
exercises the whole loop with no algorithm in the way. Real methods arrive as
`Attributor` and `Updater` implementations, not as new frameworks.

## Beyond text: typed values, steps and modalities

> **Status: design for spec version 2, not yet implemented.** Everything here is
> additive: version 1 specs, the one-file YAML and the one-agent case stay as they are.

A system is more than a chain of prompts: it renders, measures, converts files and waits
for a person. Version 2 makes the whole workflow one graph that xmas runs, traces and
optimizes, saved as a bundle that defines everything but secrets and code bodies: models
with endpoints, ids, parameters and what they accept, prompts, and tools and steps by
location. A spec that leaves any of these out cannot be rerun as it stands.
**A `.xmas` is a bundle, not one file:** a folder, zipped when shared:

```text
deck.xmas/
  xmas.yaml        # the wiring: inputs, outputs, state, nodes, edges, budget, health
  models.yaml      # the model catalogue: endpoint, model id, params, accepts, key name
  callables.yaml   # tools and steps: where they live, typed reads and writes, params
  prompts/         # one Markdown file per agent, the text its model sees
  optimize.yaml    # the task, what to tune, what is frozen
  xmas.lock        # checksums, written by the library and read by xmas verify
```

Runs are not part of it. `MAS.save(path)`, `xmas init` and optimizers (one file at a
time) write bundles; `MAS.load(path)` reads them. `xmas check` validates one without a
model call, `xmas show` draws its graph, `xmas run` runs it, `xmas diff` compares two
file by file and `xmas verify` checks its checksums. A skill is a bundle with versions;
a project pins one version, adds a models overlay and keeps its runs.

**Draw a slide, look at it, repair until it fits.** `xmas.yaml` holds the wiring and no
prompt text:

```yaml
version: 2
inputs: {spec: json}
outputs: {deck: document}
agents:
  - name: writer
    model: text-model
    for each: slide in spec.slides
    reads: [slide, verdict]  # verdict is empty on the first visit
    writes: {slide_md: text}
    tools: [web_fetch]
  - name: judge
    model: vision-model
    for each: slide in spec.slides
    reads: [slide, slide_png, measures]
    writes: {verdict: json}
steps:
  - {name: render, uses: render_slide, for each: slide in spec.slides}
  - {name: measure, uses: measure_slide, for each: slide in spec.slides}
  - {name: assemble, uses: assemble_deck}
human:
  - {name: sign_off, form: approve, reads: [deck], writes: {approved: json}}
edges:
  - writer -> render -> measure -> judge
  - judge -> writer: {when: "measures.overflow > 0 or not verdict.fits", max_visits: 3}
  - judge -> assemble: else
  - assemble -> sign_off
```

The four nodes sharing a `for each` form one chain per slide, so the back edge repairs
that slide alone; at `max_visits` the `else` edge runs and the trace marks the cap.
`assemble`, outside the chain, waits for all slides and reads per-slide values as lists.
`models.yaml` names each model once with the environment variable holding its key; this
catalogue is also the optimizer's search space for model bindings:

```yaml
text-model:
  endpoint: https://api.example.com/v1
  model: example/text-large
  params: {temperature: 0.7, max_tokens: 4000}
  accepts: [text]
  key: TEXT_MODEL_KEY  # the name of an environment variable, never the secret
vision-model:
  endpoint: https://vision.example.com/v1
  model: example/vision-small
  accepts: [text, image]
  key: VISION_MODEL_KEY
```

`callables.yaml` is the one registry for code, which it locates and never inlines:

```yaml
render_slide:
  kind: python
  import: deckkit.render:render_slide
  reads: {slide_md: text}
  writes: {slide_png: image}
  params: {width: 1920, height: 1080}
measure_slide:
  kind: python
  import: deckkit.measure:measure_slide
  reads: {slide_png: image}
  writes: {measures: json}  # overflow, smallest label, contrast; no model call
web_fetch: {kind: mcp, server: "https://tools.example.com/mcp"}
```

A callable in an agent's `tools` runs when the model decides, in a step when the graph
reaches it; the choice is topology, so an optimizer may change it. Nodes declare typed
reads and writes, checked by `xmas check` before any call, and an optimizer may add,
drop, move or swap any node that still checks; `frozen` in `optimize.yaml` opts one out.
`prompts/judge.md` may start with a header holding only its json schema:

```markdown
---
verdict:
  fits: {type: bool, means: the slide makes its claim at a glance and nothing overflows}
  problems: {type: list of text, means: one line per thing a first-time reader trips on}
  fix: {type: text, means: the one change that would help most}
---
You check one slide of a talk against what it is meant to say.
The plan for the slide: {slide}
The slide as drawn: {slide_png}
What the renderer measured: {measures}
```

xmas strips the header before the model sees the text and renders format instructions
from it that no optimizer can drop; a reply failing the schema gets one retry, then a
`bad_format` error. All else about the agent is in `xmas.yaml`, and `xmas check` fails
if the header names a write it lacks or a placeholder is not a read.

**Results in, a chart and its caption out.**

```yaml
inputs: {results: table}
outputs: {chart: image, caption: text}
agents:
  - {name: analyst, model: text-model, reads: [results], writes: {finding: json}}
  - {name: captioner, model: vision-model, reads: [chart, finding], writes: {caption: text}}
steps:
  - {name: draw, uses: draw_chart}
edges:
  - analyst -> draw -> captioner
```

The analyst's finding, whose schema heads `prompts/analyst.md`, names the kind of chart
that fits; `draw_chart`, from a chart toolbox in `callables.yaml`, draws that kind, not
one standard chart. The captioner reads the image back, so it needs a model that sees.
Its bundle: `xmas.yaml`, `models.yaml`, `callables.yaml` and two prompt files.

**From Python.** `run_state` takes typed values by name; the trace holds every output:

```python
from xmas import MAS

mas = MAS.load("deck.xmas")  # models, prompts and callables all come from the bundle
trace = mas.run_state({"spec": {"title": "Results", "slides": [...]}})
visits = [r for r in trace["judge"] if r.item == 2]  # the judge on slide 2, per visit
picture = visits[-1].inputs["slide_png"]  # the image it saw last, by reference
deck = trace.outputs["deck"]
mas.save("deck.xmas")  # rewrites only the files whose content changed
```

**Operators.** These eight are the complete set. `for each` runs a chain per list item
(consecutive nodes with the same `for each`, drawn as one box; no nested maps in version
2). `when` guards an edge; `else` is the edge taken when no `when` holds, required
wherever `when` exists. `while` is a back edge with `when` and `max_visits`. `switch` picks one edge by a value. `on error` routes by kind (money,
outage, length_cut, bad_format, ...). `first of` races branches; the first good result
wins. A box is a graph used as one node of another. Predicates: comparisons,
`and`/`or`/`not`, `exists`, `len`, `in` and the built-ins `visits`, `cost`, `elapsed`;
richer ones are registered by name. Deliberately not operators: reduce (any node reading
a list is one), filter (`for each` plus `when`), sample (a step), parallel (the default
where nothing depends) and goto (edges say where to go).

**Values.** A value is `text`, `json`, `image`, `audio`, `document` (PDF, slides, Word),
`table`, `file`, or a `list of` these; large values travel by reference (path or URI,
media type, size, checksum). Each model declares what it `accepts` and `produces` (text
by default), and binding an agent that reads an image to a model that cannot see fails
at construction. Conversions (document to text, table to rows) are visible steps;
showing a value to a model is the LLM layer's job.

**People in the graph.** Human nodes come in four forms: `ask` (a card with options),
`judge` (an A/B pair), `approve` (a gate) and `edit` (a person edits a value). The run
checkpoints and resumes on the answer; an optional `timeout` takes a default branch. The
answer is a typed value, recorded with who answered, when, what was shown and the wait.

**Records, health and budgets.** A record has its item, latency, cost, tokens, finish
reason and error kind, from which the engine derives cheap health signals with no model
call: stuck loops, empty outputs, schema failures, caps hit, cost per item, budget use,
time paused for a human. Per-run cost and wall-clock budgets end the run with a typed
stop.

**Verify.** `xmas verify` checksums every bundle file and all it references: each
callable's code (for an mcp tool, its listed schema) and each `models.yaml` entry it
binds, so anything changed since saving fails before a run. Verified skills are a later
step; the mark's form, a signature or a registry of verified bundles, is open.

**What stays the same.** Version 2 is opt-in and additive. Version 1 specs load and run
unchanged, `to_yaml`/`from_yaml` keep their one file, and a one-agent system still
encodes to its bare prompt, so promptolution's optimizers send byte-identical requests.
`run(query)` and `__call__(query) -> str` keep their signatures. Each feature above is a
field, a node kind or a check; the library stays small.

## Configuration

xmas is configured entirely through the environment. There is no config file, and no
credential is ever read from or written to the repository.

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |

Copy the template and fill it in locally:

```bash
cp .env.example .env
```

`.env` is gitignored, `.env.example` holds dummy values only, and a
[gitleaks](https://github.com/gitleaks/gitleaks) pre-commit hook scans every commit for
credentials. If you need a key in CI, add it as a repository secret and expose it as an
environment variable in the workflow. Never inline it.

Tests that need a live model run against the gateway at
[`api.tomzehle.com`](https://api.tomzehle.com), which speaks the OpenAI protocol. Point
`XMAS_BASE_URL` at it and set `XMAS_API_KEY` to your gateway key. Tests that require a
gateway are skipped when the variables are absent, so a fresh clone is green without any
credential.

## Development

```bash
uv sync --extra dev                        # create .venv and install everything
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy                               # strict type check
uv run mkdocs serve                       # docs at http://127.0.0.1:8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, commit convention and review
expectations.

## License

MIT. See [LICENSE](LICENSE).
