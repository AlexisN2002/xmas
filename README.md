# 🎄 xmas

[![CI](https://github.com/gepromptet/xmas/actions/workflows/ci.yml/badge.svg)](https://github.com/gepromptet/xmas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🎁 **xmas builds, runs and optimizes multi-agent systems.** A system is a set of agents,
the plain code steps between them, the edges that wire them together and the model behind
each agent. xmas treats all of it as one object whose parts are ordinary parameters, so an
optimizer can improve every agent's prompt, the wiring and the choice of model against a
task you care about. Every run leaves a full per-agent trace with cost, time and errors,
so you always know who was naughty and who was nice. A single agent is the smallest
system, not a special case.

Version 2 puts the whole workflow in one graph: agents, code steps, conditions and loops,
and answers with a fixed JSON shape. Pictures, documents and people are planned next.
xmas checks the wiring before the first call and draws the picture from the graph that runs.

### Key Features

- **Your first system in minutes:** a few `Agent` objects and their edges, and it runs.
- **Code and conditions in the graph:** plain Python steps, JSON answers with a fixed shape, and `when`, `while`, `switch` and `on error` edges.
- **Know what every run cost:** one record per visit with time, tokens and cost, budgets that end a run cleanly, and health signals.
- **Save and share:** a version 1 system as one YAML file today; a version 2 system as a `.mas` bundle, planned.
- **Optimization built in:** improve every agent's prompt, the wiring and the models against your task.

See [docs/design-principles.md](docs/design-principles.md) for the principles behind the
design.

> **Status: beta.** Inference, the optimization loop, CANTANTE, topology search, code steps,
> JSON answers, conditions, loops, budgets and health are in place. The `.mas` bundle,
> values beyond text, people in the graph and the experiment layer are still to come.

| What | Today |
|---|---|
| Agents, tools (local and MCP), a model per agent | built |
| Code steps between agents | built |
| Text and JSON values, JSON answers with a fixed shape | built |
| Wiring checked before the first call | built |
| `when`, `else`, `while`, `switch` and `on error` edges, merges and joins | built |
| A system inside a system (a box) | built |
| Records with cost, time and tokens, budgets, health signals | built |
| `describe()`, the system as plain data for drawing | built |
| YAML files for version 1 systems, plain-data specs for version 2 | built |
| CANTANTE, topology search, `frozen` nodes | built |
| `for each` chains and `first of` races | planned |
| Images, audio, documents, tables and files as values | planned |
| Several named inputs and outputs per run | planned |
| People in the graph, and Stop | planned |
| The `.mas` bundle, its commands and `verify()` | planned |
| Learning from human picks | planned |
| The experiment layer | planned |

## 🔧 Installation

xmas is not on PyPI yet. To use it in your own project:

```bash
uv add git+https://github.com/gepromptet/xmas
```

To work on xmas itself, clone it:

```bash
git clone https://github.com/gepromptet/xmas.git
cd xmas
uv sync
cp .env.example .env   # then set XMAS_BASE_URL and XMAS_API_KEY
```

Tools served over MCP need `uv sync --extra mcp`.

## 💡 Your first system in minutes

Two agents: an intern drafts an answer, and the boss turns it into the final one.
`{query}` and `{draft}` name what each agent reads; the intern's `output` is what the boss
receives.

```python
from xmas import MAS, APILLM, Agent

llm = APILLM(model="llama-3.1-8b")  # endpoint and key come from XMAS_BASE_URL / XMAS_API_KEY

intern = Agent("intern", prompt="Answer this question, showing your working: {query}", output="draft")
boss = Agent(
    "boss",
    prompt="Question: {query}\nYour intern's draft: {draft}\nGive the final answer in one sentence, as your own.",
    inputs=["query", "draft"],
    output="prediction",
)

mas = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)

print(mas("Why is the sky blue?"))  # the boss's answer
print(mas.last_trace["intern"][0].outputs)  # who really did the work
```

Each agent can run on its own model: give it `model="..."` and pass the models by name
through `MAS(..., llms={...})`.

## Code between agents: steps

The rest of this page grows one small system. A writer drafts, a step counts the words,
and a judge reads both. The count is plain Python, so it needs no model call.

A step runs a function from the system's `tools=` list. The function needs a type hint on
every parameter and a docstring.

```python
from xmas import MAS, Agent, Step
from xmas.tools import tool


@tool
def count_words(draft: str) -> int:
    """Count the words of a draft."""
    return len(draft.split())


writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
judge = Agent(
    "judge",
    prompt="Question: {query}\nDraft, {count_words} words: {draft}\nIs it a good answer? Say why in one sentence.",
    inputs=["query", "draft", "count_words"],
    output="review",
)
mas = MAS(
    [writer, judge],
    steps=[Step("count", uses="count_words")],
    edges=[("writer", "count"), ("count", "judge")],
    tools=[count_words],
    llm=llm,
)

print(mas("Why is the sky blue?"))  # the judge's review
print([record.agent for record in mas.last_trace.records])  # ['writer', 'count', 'judge']
```

A step reads its parameters by name, here `draft`. It writes one value named like the
function, here `count_words`. That value is text when the function returns `str`, and
JSON otherwise. `Step(..., reads={"draft": "answer"}, writes={"count_words": "words"})`
renames both sides.

The same function can also be a tool: `Agent(..., tools=["count_words"])` lets the
model call it while it answers. As a step, the graph decides when it runs; as a tool,
the model does. An MCP tool from `MCPTools` works in both places too.

In Python, an agent's reads are `inputs` and its write is `output`. A spec calls them
`reads` and `writes`.

## Answers with a shape

An agent can answer in JSON with fixed fields. The schema names each field with a type
and, optionally, what it means.

```python
verdict = {
    "fits": {"type": "bool", "means": "the draft answers the question in under 50 words"},
    "action": {"type": "text", "one of": ["publish", "shorten"]},
}
judge = Agent(
    "judge",
    prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
    inputs=["query", "draft", "count_words"],
    output="verdict",
    output_type="json",
    schema=verdict,
)
mas = MAS(
    [writer, judge],
    steps=[Step("count", uses="count_words")],
    edges=[("writer", "count"), ("count", "judge")],
    tools=[count_words],
    llm=llm,
)

trace = mas.run("Why is the sky blue?")
print(trace["judge"][0].rendered_prompt)
```

For a four-word draft, the judge's model sees this prompt. The last lines come from the
schema:

```text
Question: Why is the sky blue?
Draft, 4 words: Sunlight scatters off air.
Judge the draft.

Answer with one JSON object and nothing else. Its fields:
- fits (true or false): the draft answers the question in under 50 words
- action (one of "publish", "shorten")
```

No optimizer can drop these format lines. An answer that does not fit gets exactly one
retry, with a note that says what was wrong. The record keeps the refused answer:

```python
print(trace.output)  # {"fits": true, "action": "publish"}
print(trace["judge"][0].rejected)  # the answer refused before the retry, if there was one
```

Two misses in a row end the run. Then `trace.status` is `failed` and `trace.error_kind`
is `bad_format`; calling `mas(...)` raises a `RuntimeError` instead.

Each field takes a `type`: `text`, `bool`, `int`, `number`, `json`, `object` or
`list of <type>`. It may add `means`, `optional`, `one of`, and `fields` for an object or
a list of objects.

## Mistakes found before any call

xmas checks the wiring when you build the system, before any model call. It lists every
problem at once, so one edit fixes them all. This judge has three:

```python
judge_with_typos = Agent(
    "judge",
    prompt="Draft: {draft}\nIt has {count_words} words. The style guide: {style}",
    inputs=["draft", "notes"],
    output="verdict",
)
try:
    MAS(
        [writer, judge_with_typos],
        steps=[Step("count", uses="count_word")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )
except ValueError as problems:
    print(problems)
```

The message names each one:

```text
The system does not check (3 problems):
- agent 'judge': prompt placeholders ['count_words', 'style'] are not among its reads ['draft', 'notes']
- Step 'count' uses unknown callable 'count_word'; available: ['count_words'].
- 'judge' reads 'notes', which is neither an input nor written by any node
```

The check also refuses two writers of one value, a JSON value sent to a function that
wants text, and a condition that does not parse.

## Conditions and loops

An edge can carry a condition. Here the judge sends the draft back while it does not fit,
then publishes it or says sorry. Two more steps do the publishing and the apology:

```python
from xmas import Edge, EdgeKind


@tool
def publish(draft: str) -> str:
    """Publish a draft."""
    return f"Published: {draft}"


@tool
def apologize(query: str) -> str:
    """Say sorry when no draft fits."""
    return f"Sorry, no short answer to: {query}"
```

The writer now reads the judge's last verdict, which is empty on its first visit. The
edges say where the run goes after each node:

```python
writer = Agent(
    "writer",
    prompt="Write a short answer to: {query}\nThe judge's last verdict, empty at first: {verdict}",
    inputs=["query", "verdict"],
    output="draft",
)
mas = MAS(
    [writer, judge],
    steps=[Step("count", uses="count_words"), Step("publish", uses="publish"), Step("sorry", uses="apologize")],
    edges=[
        ("writer", "count"),
        ("count", "judge"),
        Edge("judge", "writer", kind=EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
        Edge("judge", "publish", kind=EdgeKind.SWITCH, switch="verdict.action"),
        Edge("judge", "sorry", kind=EdgeKind.ELSE),
        Edge("count", "sorry", kind=EdgeKind.ERROR, errors=("timeout", "outage")),
    ],
    tools=[count_words, publish, apologize],
    llm=llm,
)

print(mas("Why is the sky blue?"))  # "Published: ..." or "Sorry, no short answer to: ..."
```

xmas decides on the judge's fresh verdict. While `fits` is false, the run goes back to
the writer. After three writer visits the loop stops, the record says `capped`, and the
`else` edge runs. When the draft fits, the `switch` edge whose name equals
`verdict.action` is taken. Any other action falls through to `else`. If counting fails
with a timeout or an outage, the run goes on at `sorry` and keeps the failed record.

These are the operators. A spec writes each edge as `a -> b`, with its operator after a
colon:

| Operator | Python | In a spec | Today |
|---|---|---|---|
| `when` | `Edge("a", "b", kind=EdgeKind.WHEN, when="...")` | `a -> b: {when: ...}` | built |
| `else` | `Edge("a", "b", kind=EdgeKind.ELSE)` | `a -> b: else` | built |
| `while` | `Edge("a", "b", kind=EdgeKind.WHILE, when="...", max_visits=3)` | `a -> b: {when: ..., max_visits: 3}` | built |
| `switch` | `Edge("a", "b", kind=EdgeKind.SWITCH, switch="verdict.action")` | `a -> b: {switch: verdict.action}` | built |
| `on error` | `Edge("a", "b", kind=EdgeKind.ERROR, errors=("timeout",))` | `a -> b: {on error: [timeout]}` | built |
| box | `Box(name, system, reads=..., writes=...)` | `boxes:` | built |
| `for each` | | | planned |
| `first of` | | | planned |

A `switch` edge's case is its target's name unless `case="..."` says otherwise. An
`on error` edge takes kinds such as `timeout`, `outage`, `rate_limited`, `money`,
`length_cut`, `bad_format` and `refusal`, or `("any",)`. A budget stop is never caught.

A condition is a small language, parsed when the system is built and never run as code.
It knows paths such as `verdict.fits`, numbers, quoted strings, `true`, `false` and
`null`. It compares with `== != < <= > >= in` and combines with `and`, `or` and `not`.
Its built-ins are `exists(path)`, `len(x)`, `visits`, `visits(node)`, `cost` and
`elapsed`. `verdict.fits = true` is refused with the position of the stray `=`.

Some things are deliberately not operators. A reduce is any node that reads a list. A
filter is `for each` plus `when`. A sample is a step. Parallel is the default wherever
nothing depends on anything else. A goto is what an edge already is.

A node fed by several edges runs once, after all of them have arrived: a join. When its
edges can never arrive in the same pass, as with `sorry` above, it runs on whichever
arrives: a merge. The check refuses any other mix and names the edges that clash.

## Reading a run

`mas.run(query)` returns the trace; `mas(query)` returns the answer and keeps the trace in
`mas.last_trace`. Suppose the judge never accepts a draft. The trace then shows the loop
and where it stopped:

```python
trace = mas.run("Why is the sky blue?")
print(trace.status)  # ok
print([record.agent for record in trace.records])
# ['writer', 'count', 'judge', 'writer', 'count', 'judge', 'writer', 'count', 'judge', 'sorry']
print(trace["judge"][-1].capped)  # True
print(trace.health.loops)  # {'judge -> writer': {'taken': 2, 'cap_hit': True}}
```

`trace["judge"]` is one record per visit. Each record holds the `inputs` it read, the
`outputs` it wrote and the `rendered_prompt` it sent. It also holds `latency_s`,
`cost_usd`, `tokens`, `finish_reason`, `error_kind` and `served_model`, the model that
actually answered. For a node with conditional edges, `route` names where the visit sent
the run. A step's record has kind `step`.

A run ends with a status: `ok`, `failed` or `over_budget`. A failed run keeps
every record that finished before the failure, and `trace.error_kind` says what kind it
was. Cost is never guessed as zero. It comes from the endpoint's response, or from a
price you give the model (`APILLM(price=TokenPrice(...))`, from `xmas.llms.meta`), or
it stays `None`. A sum with one unknown part is unknown too, so `trace.cost_usd` is
`None` when any record's cost is. A step costs `0.0`, since it makes no model call.

### Health

Every trace carries health signals, read from the records with no model call. They are
information, never rules. Suppose the writer also sent the same draft each time:

```python
print(sorted({(flag.signal, flag.node) for flag in trace.health.flags}))
# [('stuck', 'count'), ('stuck', 'judge'), ('stuck', 'writer')]
```

The signals are `retried`, `bad_format`, `error_routed`, `empty`, `stuck` (the same
output as the node's last visit), `unfilled` (a prompt placeholder with no value), `leak`
(chat-template tokens or reasoning tags in an output), `long` and `short`.
`trace.health.loops` counts each `while` edge, and `trace.health.budget` shows each limit
next to what the run spent. `MAS(..., health=HealthSettings(length_factor=4))` tunes them.
A healthy run's `trace.to_dict()` has no `health` key.

### Budgets

A budget limits every run, and every item of a batch, on its own. A call starts only
while the run is below every limit. This copy of the quickstart may make one call:

```python
from xmas import Budget

thrifty = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_calls=1))
spent = thrifty.run("Why is the sky blue?")
print(spent.status)  # over_budget
print(spent.error)  # BudgetExceeded: calls 1 >= max 1 before call 2
print(spent["intern"][0].outputs)  # the intern's draft is kept
```

`Budget` also takes `max_cost_usd` and `max_wall_s`. A limit of 0 spends nothing. Calls
already in flight may overshoot by one call per branch. A cost limit refuses the next
call when the last one reported no cost and its model has no price, because an unknown
cost cannot be checked.

## Drawing a system

`mas.describe()` returns the system as plain data. Each node has its kind, model, typed
reads and writes, and whether it is frozen. Each edge has its kind and condition:

```python
picture = mas.describe()
print([(node["name"], node["kind"]) for node in picture["nodes"]])
# [('writer', 'agent'), ('count', 'step'), ('judge', 'agent'), ('publish', 'step'), ('sorry', 'step')]
print([edge["kind"] for edge in picture["edges"] if not edge.get("implicit")])
# ['plain', 'plain', 'while', 'switch', 'else', 'error']
mas.agents["judge"].prompt += "\nBe strict."
print(mas.describe()["id"] == picture["id"])  # True
```

The `id` is a hash of the description, and prompts are not part of it. So editing a
prompt keeps the id, and changing the wiring changes it. Any picture of the system comes
from the graph that runs, never from a hand-drawn copy.

## A system inside a system

A box runs a whole system as one node of another. Its `reads` map the inner inputs to
outer values, and its `writes` map inner values back out:

```python
from xmas import Box

cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
shorten = Box(
    "shorten",
    MAS([cutter], inputs={"page": "text"}, llm=llm),
    reads={"page": "draft"},
    writes={"short": "summary"},
)
first_draft = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
outer = MAS([first_draft], boxes=[shorten], edges=[("writer", "shorten")], llm=llm)

trace = outer.run("Why is the sky blue?")
print(trace.output)  # the cutter's shorter draft
print([(path, record.agent) for path, record in trace.walk()])
# [((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]
```

The box spends from the outer run's budget. Its record has kind `box` and holds the
inner trace, and `trace.walk()` yields every record with the path of boxes around it.
The box's cost and tokens are its inner records' sums, unknown when any of them is, and
a failed box keeps the inner records that finished.

## 💾 Save and share

A version 1 system, such as the quickstart's, saves as one YAML file. One file holds
the whole system: prompts, wiring and models, never keys or endpoints.

```python
team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)  # the quickstart's system
team.to_yaml("team.yaml")
team = MAS.from_yaml("team.yaml", llm=llm)
```

A system that uses steps, JSON answers, conditions or boxes is version 2. Its spec is
plain data, and `MAS.from_dict` rebuilds it with the same models and functions:

```python
spec = mas.to_dict()  # version, agents, steps, edges, callables, prompts
rebuilt = MAS.from_dict(spec, llm=llm, tools=[count_words, publish, apologize])
print(rebuilt.describe()["id"] == mas.describe()["id"])  # True
```

The spec's `edges` read like this in YAML, and `from_dict` accepts them in this form:

```yaml
- writer -> count -> judge
- judge -> writer: {when: not verdict.fits, max_visits: 3}
- judge -> publish: {switch: verdict.action}
- judge -> sorry: else
- count -> sorry: {on error: [timeout, outage]}
```

`callables` records where each function lives and what it reads and writes. Models
appear by name only. Each agent's prompt sits under `prompts` for now; in the planned
bundle, each prompt gets a file of its own. `to_yaml` refuses a version 2 system, since
its file format is the `.mas` bundle (see below), and points to `to_dict()` until then.
Every version 1 file loads, runs and saves as before.

## 🧠 Optimization built in

CANTANTE improves every agent's prompt at once, splitting each score into credit per agent.
Here it tunes the quickstart's two agents, `team` from the section above.

```python
from promptolution.optimizers import OPRO
from xmas import GSM8KTask
from xmas.optimization import CantanteOptimizer, ContrastiveAttributor

task = GSM8KTask.from_huggingface(split="train", n=50)  # needs: uv run --with datasets
optimizer = CantanteOptimizer(team, task, ContrastiveAttributor(llm), OPRO, optimizer_kwargs={"meta_llm": llm})
best = optimizer.optimize(n_steps=5)
best.to_yaml("team-optimized.yaml")
```

`MAS(..., frozen=("boss",))` keeps a node as it is. No optimizer changes its prompt,
model, reads, writes or edges. A frozen node makes the system version 2, so keep its
`to_dict()` instead of a YAML file. `EvolutionaryTopologyOptimizer` searches the wiring. It
keeps steps, boxes, frozen nodes and conditional edges as they are, and drops any
candidate the check refuses. Steps are never part of the prompts an optimizer sees.

## 🔗 xmas and promptolution

xmas builds on [promptolution](https://github.com/gepromptet/promptolution): a system is a
promptolution predictor and a task is a promptolution task, so promptolution's optimizers
run on any agent unchanged.

A one-agent system encodes to its bare prompt, with or without steps. So promptolution's
optimizers send the same requests to it as to a single prompt.

## Where this is going

> **Planned, not in the code yet.** Nothing in this section runs today. The code below
> shows how it will read.

### One bundle per system: `name.mas/`

A version 2 system will save as a bundle, a folder in which each part has its own file.
No prompt text sits in the wiring:

```text
team.mas/
  xmas.yaml        # the wiring: inputs, outputs, nodes, edges, budget, health
  callables.yaml   # where each tool and step lives, with typed reads and writes
  prompts/         # one plain-text file per agent
  optimize.yaml    # the task, what to tune, what stays frozen
```

The folder also carries a few warm-up examples and a lock file of checksums. Runs are
never part of it. The running example's `xmas.yaml` would read:

```yaml
version: 2
inputs: {query: text}
agents:
  - {name: writer, model: text-model, reads: [query, verdict], writes: {draft: text}}
  - {name: judge, model: text-model, reads: [query, draft, count_words], writes: {verdict: json}}
steps:
  - {name: count, uses: count_words}
  - {name: publish, uses: publish}
  - {name: sorry, uses: apologize}
edges:
  - writer -> count -> judge
  - judge -> writer: {when: not verdict.fits, max_visits: 3}
  - judge -> publish: {switch: verdict.action}
  - judge -> sorry: else
  - count -> sorry: {on error: [timeout, outage]}
```

A prompt file is plain text. It may start with a short header that holds only the
answer's schema. xmas strips the header before the model sees the text. `prompts/judge.md`:

```markdown
---
verdict:
  fits: {type: bool, means: the draft answers the question in under 50 words}
  action: {type: text, one of: [publish, shorten]}
---
Question: {query}
Draft, {count_words} words: {draft}
Judge the draft.
```

`callables.yaml` locates code and never inlines it:

```yaml
count_words:
  kind: python
  import: mytools.text:count_words
  reads: {draft: text}
  writes: {count_words: json}
```

The folder names models and tools. Addresses and keys live in your configuration
(`XMAS_BASE_URL`, `XMAS_API_KEY`), never in the folder. A model name the configured
endpoint does not know gives a warning with the closest names from its catalogue. An
MCP tool names its server; the server's address sits in your configuration too.

From Python and from the shell, it will read like this:

```python
mas = MAS.load("team.mas")
mas.verify()
mas.save("team.mas")  # rewrites only the files whose content changed
```

```bash
xmas init team.mas      # a new folder
xmas check team.mas     # the wiring check, no model call
xmas show team.mas      # the graph as a picture
xmas run team.mas "Why is the sky blue?"
xmas diff a.mas b.mas   # file by file
xmas verify team.mas
```

`verify()` works on three levels. First, checksums: every file, and the code it points
to, is unchanged since it was saved. Second, a dry check of the wiring, the prompts and
the model names at the endpoint, with no model call. Third, the warm-up examples run
under a small budget: a step must match exactly, and an agent must pass the example's
check. A failure warns by default. Run it after a model switch, after an optimizer run
and before a release.

### Beyond text

Values today are `text` and `json`. Planned next are `image`, `audio`, `document`,
`table`, `file` and `list of` each. The check already knows these words and refuses
them with a message. Large values travel by reference. Each model states what it
accepts, and an agent that reads an image on a model that cannot see fails when the
system is built. Conversions, such as a document to text, are visible steps.

`for each` runs a chain once per item of a list. Take a deck: a writer drafts a slide, a
step renders it, and a judge looks at the picture. The three share one `for each`, so
the `while` edge repairs that one slide only. At the cap, the `else` edge runs and the
trace marks it. A node outside the chain, such as one that assembles the deck, waits
for every slide and reads the results as lists:

```yaml
agents:
  - {name: writer, model: text-model, for each: slide in spec.slides, reads: [slide, verdict], writes: {slide_md: text}}
  - {name: judge, model: vision-model, for each: slide in spec.slides, reads: [slide, slide_png], writes: {verdict: json}}
steps:
  - {name: render, uses: render_slide, for each: slide in spec.slides}
  - {name: assemble, uses: assemble_deck}
edges:
  - writer -> render -> judge
  - judge -> writer: {when: not verdict.fits, max_visits: 3}
  - judge -> assemble: else
```

`first of` races branches, and the first good result wins. `mas.run_state({...})` starts
a run from several named values and returns several outputs.

### People in the graph

A person becomes a node with one of four forms. `ask` shows a card with options,
`judge` shows a pair to pick from, `approve` is a gate, and `edit` lets a person change
a value. The run pauses with a checkpoint and resumes on the answer, after a restart
too. An optional timeout takes a default edge. The trace records who answered, when,
what they were shown and how long the run waited. A Stop ends a running system within
seconds and keeps every record so far.

A preference task will learn from those pairwise picks, with credit per node, so an
optimizer can improve a system against what people chose.

## 📚 Scientific Publications

Methods implemented in xmas:

- **CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — Zehle, 2026. [arXiv](https://arxiv.org/abs/2605.13295) · `CantanteOptimizer`, `ContrastiveAttributor`

xmas builds on:

- **promptolution: A Unified, Modular Framework for Prompt Optimization** — Zehle et al., 2026. [EACL 2026](https://aclanthology.org/2026.eacl-demo.21/)
- **CAPO: Cost-Aware Prompt Optimization** — Zehle et al., 2025. [AutoML 2025](https://proceedings.mlr.press/v293/zehle25a.html)

## 🏗 Components

- **`Agent`** – One node: a prompt template, the values it reads and the value it writes.
- **`Step`** – A node that runs a function or an MCP tool from the system's `tools=`, with no model call.
- **`Edge`** – One edge with its operator, an `EdgeKind`: plain, `when`, `else`, `while`, `switch` or `on error`.
- **`Box`** – A whole system used as one node of another.
- **`MAS`** – The system and the predictor: agents, steps, edges, models and tools; runs a query and keeps its trace.
- **`Trace`** – The per-agent record of one run: rendered prompts, outputs, tokens, cost, time, tool calls, status.
- **`Budget`** – Limits on calls, cost and wall time for each run.
- **`Health`** – Signals read from a finished run: stuck loops, empty answers, retries, caps hit.
- **`MASTask`** – A dataset plus a reward over the system's answers, e.g. `GSM8KTask`.
- **`APILLM`** – Any OpenAI-compatible endpoint, with scoring, tool turns and a per-call log.
- **`MASOptimizer`** – Propose, evaluate through the task, select; CANTANTE and topology search build on it.

## Configuration

xmas is configured entirely through the environment. There is no config file, and no
credential is ever read from or written to the repository. A saved system never holds
an endpoint or a key.

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |

## Development

```bash
uv sync                                    # create .venv and install everything
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy                               # strict type check
uv run mkdocs serve                       # docs at http://127.0.0.1:8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, commit convention and review
expectations. See [CHANGES.md](CHANGES.md) for what changed in version 2.

## 📄 License

See [LICENSE](LICENSE).
