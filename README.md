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
chains that run once per item of a list, answers with a fixed JSON shape, pictures,
documents and other files passed by reference, people who approve or edit, branches that
race, and a Stop that stops. A system saves as a `.mas` bundle, a folder of plain files, and
checks itself with a few warm-up examples.
xmas checks the wiring before the first call and draws the picture from the graph that runs.
A system made only of agents and plain edges is a version 1 system, and works as before.

### Key Features

- **Your first system in minutes:** a few `Agent` objects and their edges, and it runs.
- **Code and conditions in the graph:** plain Python steps, JSON answers with a fixed shape, and `when`, `while`, `switch` and `on error` edges.
- **Pictures and files as values:** a model that can see gets the pictures, and the check refuses one that cannot.
- **People in the graph:** a person approves, picks, chooses or edits; the run pauses, and resumes after a restart too.
- **Know what every run cost:** one record per visit with time, tokens and cost, budgets that end a run cleanly, and health signals.
- **Save and share:** a system as a `.mas` bundle with one file per prompt, checked by `verify()` and the `xmas` command.
- **Optimization built in:** improve every agent's prompt, the wiring and the models against your task.

See [docs/design-principles.md](docs/design-principles.md) for the principles behind the
design.

> **Status: beta.** Inference, the optimization loop, CANTANTE, topology search, code steps,
> JSON answers, conditions, loops, per-item chains, media values, budgets, health, people in
> the graph, Stop, `first of` races, the `.mas` bundle, `verify()` and the `xmas` command are
> in place. Learning from human picks and the experiment layer are still to come.

| What | Today |
|---|---|
| Agents, tools (local and MCP), a model per agent | built |
| Code steps between agents | built |
| Text and JSON values and lists of them, JSON answers with a fixed shape | built |
| Wiring checked before the first call | built |
| `when`, `else`, `while`, `switch` and `on error` edges, merges and joins | built |
| A system inside a system (a box) | built |
| Records with cost, time and tokens, budgets, health signals, per item too | built |
| `describe()`, the system as plain data for drawing | built |
| YAML files for version 1 systems, plain-data specs for version 2 | built |
| Optimizers: CANTANTE, topology search, `frozen` nodes | built |
| `for each` chains: some nodes once per item of a list | built |
| `run_state`: start a run from several named values and read several outputs | built |
| Images, audio, documents, tables and files as values, passed by reference | built |
| Models that state what they accept and produce, checked when the system is built | built |
| `first of` races: the first good branch wins | built |
| People in the graph: pauses, answers, resume after a restart | built |
| Stop: a running system ends within moments and keeps its records | built |
| The `.mas` bundle: save, load, a lock of checksums | built |
| `verify()`: checksums, a dry check, warm-up examples | built |
| Configuration: endpoint, key, catalogue and tool servers, outside the bundle | built |
| The `xmas` command: `init`, `check`, `show`, `run`, `diff`, `verify` | built |
| Learning from human picks | planned |
| The experiment layer: one config for a reproducible run, like promptolution's `ExperimentConfig` | planned |

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

xmas reads the two variables from the environment and never opens `.env` itself.
Export them, or start Python with `uv run --env-file .env python`. A configuration file
works too (see Configuration). Installing xmas also installs the `xmas` command.

Tools served over MCP need `uv sync --extra mcp`.

## 💡 Your first system in minutes

Two agents: an intern drafts an answer, and the boss turns it into the final one.
The question you pass in is the value `query`, which an agent reads unless `inputs=`
says otherwise. `{query}` and `{draft}` mark where each value goes in a prompt. The
intern's `output`, `draft`, is what the boss receives.

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
print(mas.last_trace["intern"][0].outputs)  # {'draft': ...}: who really did the work
```

The model name must be one your endpoint serves. Each agent can run on its own model:
give it `model="..."` and pass the models by name through `MAS(..., llms={...})`.

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

In Python, an agent's reads are `inputs` and its write is `output`. The saved spec (see
Save and share) calls them `reads` and `writes`.

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
| `for each` | `MAS(..., each={"b": "page in plan.pages"})` | `for each: page in plan.pages` on each node | built |
| `first of` | `Edge("a", "b", kind=EdgeKind.FIRST)` | `a -> b: first of` | built |

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
(chat-template tokens or reasoning tags in an output), and `long` and `short` (over
three times, or under a third of, the node's median length). A `for each` chain adds
`item_failed` and `item_over_budget` (see Once per item).
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
already in flight may overshoot by one call per branch. `max_parallel_calls` caps how
many model calls are in flight at once, across branches, boxes and chain items. A cost
limit refuses the next call when the last one reported no cost and its model has no
price, because an unknown cost cannot be checked.

## Drawing a system

`mas.describe()` returns the system as plain data. Each node has its kind, model, typed
reads and writes, and whether it is frozen. Each edge has its kind and condition. The
start and end wiring that xmas adds itself is marked `implicit`:

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
outer values, and its `writes` map inner values back out. The inner system names its one
input `page` with `inputs=`; without it, a system's input is `query`:

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

## Once per item: `for each`

A `for each` chain runs some nodes once per item of a list. Here a planner lists the
pages of an answer in JSON. Each page is written and counted on its own, and a step
outside the chain adds up the counts:

```python
from xmas import ForEach


@tool
def total_words(count_words: list[int | None]) -> int:
    """Add up the word counts of the pages that were written."""
    return sum(count for count in count_words if count is not None)


planner = Agent(
    "planner",
    prompt="Plan a short answer in three pages: {query}",
    output="plan",
    output_type="json",
    schema={"pages": {"type": "list of object", "fields": {"id": {"type": "text"}, "topic": {"type": "text"}}}},
)
page_writer = Agent("page_writer", prompt="Write one page on: {page}", inputs=["page"], output="draft")
deck = MAS(
    [planner, page_writer],
    steps=[Step("count", uses="count_words"), Step("total", uses="total_words")],
    edges=[("planner", "page_writer"), ("page_writer", "count"), ("count", "total")],
    each={"page_writer": ForEach("page", "plan.pages", at_once=2, key="id"), "count": "page in plan.pages"},
    tools=[count_words, total_words],
    llm=llm,
)

trace = deck.run("Why is the sky blue?")
print([(record.agent, record.item_key) for record in trace.records])
# [('planner', None), ('page_writer', 'light'), ('count', 'light'), ('page_writer', 'air'), ...]
print(trace["total"][0].inputs)  # {'count_words': '[4, null, 4]'}: the page 'air' failed
```

Every node of the chain names the same item and list. The first one may add options:
`at_once` caps how many items run at the same time, and `key` names the field that
tells the items apart in the records.

Inside the chain, `page` is one item and `count_words` one number. Outside, each value
the chain writes is a list in item order, however the items finish. When one item
fails, say its model call times out, its place in each list is `None` and the other
items go on. The run still ends `ok`, and health counts the failed item:

```python
print(trace.health.items["page_writer"]["failed"])  # 1, of 3 items
print([flag.signal for flag in trace.health.flags])  # ['item_failed']
```

A budget stop or an out-of-money error still ends the whole run. `item_budget={"cost_usd":
0.05}` on the `ForEach` flags an item that costs more as `item_over_budget`, and never
stops it. A `while` edge inside a chain repairs one item, and its `max_visits` counts per
item. In a spec, each node of the chain carries `for each: page in plan.pages`; the
first may write `{item, in, at once, key, item budget}` instead. Chains do not nest.

## Several values in, several out: `run_state`

`run(query)` takes one text. `run_state` starts a run from all of a system's inputs by
name, and `outputs=` names the values the trace hands back. Here are the pages again,
started from a plan you already have:

```python
pages = MAS(
    [page_writer],
    steps=[Step("count", uses="count_words"), Step("total", uses="total_words")],
    edges=[("page_writer", "count"), ("count", "total")],
    each={"page_writer": ForEach("page", "plan.pages", key="id"), "count": "page in plan.pages"},
    inputs={"plan": "json"},
    outputs={"total_words": "json", "count_words": "list of json"},
    tools=[count_words, total_words],
    llm=llm,
)

heard = []
trace = pages.run_state({"plan": {"pages": [{"id": "light"}, {"id": "eye"}]}}, on_event=heard.append)
print(trace.outputs)  # {'total_words': 8, 'count_words': [4, 4]}
print(trace.output)  # 8: the first output, which a task scores
print(sorted({event.kind for event in heard}))  # ['end', 'item_end', 'item_start', 'start']
```

A missing, extra or mistyped input raises one `ValueError` that lists them all, before
any call. `on_event` hears each visit's start and end and each item's, with the node,
the item and a status. It runs on the run's threads, so keep it quick.

With `outputs=`, several branches may end the run, and the run ends once all of them
have. Without it, `output` is the last terminal's write, as before. The trace of a
version 2 run carries a `run_id`, and its `graph_id` is the system's `describe()["id"]`.

Three more things a node can do. `MAS(..., retry={"count": {"outage": 1}})` runs a node
again after an outage, a timeout or a rate limit. Its one record counts the `retries`
and the cost of every attempt. A node may read two values that no node writes: `error`,
the `{kind, message, node}` of the failure an `on error` edge routed to it, and `run`,
what the run has spent so far. `Agent(..., inputs={"slide": "page"})` fills the
placeholder `{slide}` from the value `page`.

## Pictures, documents and other files

A value can also be an `image`, `audio`, a `document`, a `table` or a `file`, or a list of
them. Such a value is a reference, a `Ref`: where the file is, its media type, its size
and a checksum. The bytes stay in the file until a model needs them.

Here a reviewer looks at the pages of a deck, and a step keeps the first page as the
cover. The pages come in as files on disk:

```python
from pathlib import Path
from xmas import MAS, APILLM, Agent, Image, Step
from xmas.tools import tool


@tool(writes={"cover": "image"})
def keep_cover(pages: list[Image]) -> Path:
    """Keep the first page as the deck's cover."""
    return Path(pages[0].location)


reviewer = Agent("reviewer", prompt="Review these slides.\n{pages}", inputs=["pages"], output="review")
vision = APILLM(model="vision-model", accepts=("text", "image"))
deck = MAS(
    [reviewer],
    steps=[Step("cover", uses="keep_cover")],
    inputs={"pages": "list of image"},
    outputs={"review": "text", "cover": "image"},
    tools=[keep_cover],
    llm=vision,
)

trace = deck.run_state({"pages": [Path("slide-1.png"), Path("slide-2.png")]})
print(trace["reviewer"][0].rendered_prompt)
print(trace.outputs["cover"].location)  # a copy in the store, named by its checksum
```

The model gets each page as a picture between the parts of the prompt's text. Each item
of a list gets a label, counted from 0 like `item`, so the model can name a page. The
record shows a marker in place of each picture, never its bytes:

```text
Review these slides.

[pages 0] [image slide-1.png, image/png, 182 KB]
[pages 1] [image slide-2.png, image/png, 96 KB]
```

`run_state` turns each `Path` into a reference to your file. The checksum is checked on
every read, so a file that changes during the run is refused. A step that returns a
`Path` writes a copy in the store instead, so a step that rewrites one file per item
leaves earlier values alone. A step may also return a `Ref`. Bytes are refused, since
`Ref.from_bytes(data, "image/png")` stores them. The store is a folder (see
Configuration), and the same bytes are stored once. In a trace, a reference is a small
dictionary, so `trace.to_dict()` stays plain JSON.

Each model states what it accepts and produces. On a model that takes only text, such
as the quickstart's `llm`, the check refuses the reviewer before any call with a
`ValueError`:

```python
MAS([reviewer], inputs={"pages": "list of image"}, llm=llm)
```

```text
The system does not check (1 problem):
- agent 'reviewer' reads image 'pages', but the default model accepts only text; give it accepts = ("text", "image") or convert it first with a step
```

What a model takes comes from the first of three places: the system's `models` entry for
the model's name, the model's own `accepts` and `produces`, or else text only. In a spec,
the `models` section names each model and says what it accepts and produces:

```yaml
models:
  vision-model: {accepts: [image]}
  painter-model: {produces: [image]}
```

`MAS(..., models={"vision-model": {"accepts": ["image"]}})` says the same in Python.
An entry may also set model parameters (see Save and share). Any other key is refused,
and so is a name no agent binds.
The rest of the rules:

- A model sees an image or audio as a content part. `APILLM` sends parts; another
  model class needs `generate_parts_with_meta` and `full_response = True`.
- A model never reads a document or a table directly. A step turns it into text or rows
  first, so the conversion shows in the graph. `xmas.convert` has two such functions:
  `document_text` reads text, Markdown and HTML, and `table_rows` reads CSV and TSV. For
  a PDF, write a step of your own.
- An agent with `output_type="image"` answers with a picture, on a model that produces
  images. The picture goes to the store, and the value is its reference. Such an agent
  takes no tools.
- An MCP tool may write a picture too: `MCPTools(..., writes={"paint": {"png": "image"}})`.
  A remote tool reads no media, so the check refuses a step that passes it one.
- A tool an agent calls reads no media either: the model would make up the reference.
  Run such a function as a step.

## People in the graph

A person can be a node too. A `Human` asks a question, and the run waits for the answer.
Here an editor approves the draft before it is published, with the draft in the question:

```python
from xmas import END, Human

editor = Human(
    "editor",
    "approve",
    reads=["draft"],
    writes="approval",
    question="Publish this answer? {draft}",
    labels={"yes": "Publish", "no": "Hold it"},
)
gated = MAS(
    [Agent("writer", prompt="Write a short answer to: {query}", output="draft")],
    steps=[Step("publish", uses="publish")],
    humans=[editor],
    edges=[
        ("writer", "editor"),
        Edge("editor", "publish", kind=EdgeKind.WHEN, when="approval.approved"),
        Edge("editor", END, kind=EdgeKind.ELSE),
    ],
    tools=[publish],
    llm=llm,
)
```

Run it under a `run_id` of your choice. It returns paused, and the answer resumes it:

```python
trace = gated.run_state({"query": "Why is the sky blue?"}, run_id="sky-1")
print(trace.status, [question.id for question in trace.pause.questions])  # paused ['sky-1/editor/1']
print(trace.pause.questions[0].options)  # ('Publish', 'Hold it')
done = gated.resume("sky-1", {"sky-1/editor/1": {"approved": True, "who": "tom"}})
print(done.status, done.output)  # ok Published: Sunlight scatters off air.
```

The run pauses only when nothing else is left to do. Other branches run on and end
first, and all open questions are asked together in `trace.pause`. `gated.answer(run_id,
answers)` keeps answers while the run still goes. `gated.resume(run_id, answers)` goes on
from the pause. Each answer is checked against its question, and a wrong one is refused
with every problem listed.

A person answers in one of four forms:

| Form | Shows | Answers |
|---|---|---|
| `ask` | `options`: a list of labels, or the path of a list the node reads | `{option}` |
| `judge` | `pair`: two values the node reads | `{pick}`: one of the two names, or `"tie"` |
| `approve` | `labels`: the yes and no choices | `{approved, option}` |
| `edit` | `edits`: the value to change | `{changed}`, and the new value under `writes={"answer": ..., "value": ...}` |

Every answer also holds `who`, `asked_at`, `answered_at`, `comment` and `timed_out`. The
question and the labels may name the node's reads, paths too, such as `{plan.title}`. The
visit keeps a record of kind `human`: what was shown, the answer as `raw_response`, a cost
of 0 and the wait in `wait_s`. An edited value must have the type of the value it
replaces, and a host's `answered_at` is an ISO time with its zone.

A pause outlives the process when the run store is a file: `MAS(..., store="runs.db")`,
or the variable `XMAS_RUNS`. A new process builds the same system and calls `resume`. A
changed system is refused. The budget counts on from the pause, and the wait is not wall
time. By default the store is in memory, so a run resumes in the same process only.

`timeout_s` gives a question a deadline, and `default` is the answer then taken, with
`timed_out` true. It routes like any other answer. A host passes `asker=`: an object
whose `ask(question)` answers at once or returns `None`. It may also have
`withdraw(question, reason)`, called when a paused run is stopped. `run`, `run_batch` and
the optimizers never wait: a question the asker leaves open takes its default, or the run
fails and names the node, from inside a chain item or a box too. `ReplayAsker(traces)` answers from recorded runs, so an
optimizer can rerun a system with people in it.

## Stop

`mas.stop(run_id)` ends a running run from another thread. No call, step or item starts
after it, and a model call in flight is cut. Here Stop is pressed one second into a long
essay:

```python
import threading

essay = MAS(
    [Agent("writer", prompt="Write a long essay on: {query}", output="draft")],
    outputs={"draft": "text"},
    llm=llm,
)
threading.Timer(1.0, essay.stop, args=("essay-1",)).start()
trace = essay.run_state({"query": "Why is the sky blue?"}, run_id="essay-1")
print(trace.status, trace.error)  # stopped Stopped: stopped
```

The trace comes back within moments of the Stop and keeps every record so far. A visit
cut after a call keeps a failed record with what its calls used. No `on error` edge
catches a stop. A paused run is marked stopped, and its open questions are withdrawn,
as are those of a run stopped while it runs.

A Python step, a promptolution batch call or an MCP tool call is not cut. The run ends
at the next boundary after it, and a step may poll `xmas.cancel_flag()` to end early.
Streaming is opt-in per model: `APILLM(..., stream=True)` streams each answer, so a cut
call keeps the text received so far as its `raw_response`. Without it, `APILLM` sends the
same request as before.

## Racing branches: `first of`

A `first of` race starts several branches from one node, and the first good result
wins. Here a quick and a careful writer race to write the draft from one outline:

```python
outline = Agent("outline", prompt="Outline an answer to: {query}", output="outline")
quick = Agent("quick", prompt="One line from: {outline}", inputs=["outline"], output="draft")
careful = Agent("careful", prompt="A careful answer from: {outline}", inputs=["outline"], output="draft")
racing = MAS(
    [outline, quick, careful],
    edges=[Edge("outline", "quick", kind=EdgeKind.FIRST), Edge("outline", "careful", kind=EdgeKind.FIRST)],
    llm=llm,
)

trace = racing.run("Why is the sky blue?")
print(sorted(record.agent for record in trace.records if record.lost))  # ['careful']
print(trace.output)  # the quick draft
```

All `first of` edges of one node form one race. The winner's writes stand. The other
calls are cut, and each loser keeps a record with `lost` set and what it paid. A step
that races cannot be cut: it runs to its end, and its writes are dropped. Contestants
write the same values and go on to the same node, which runs once, or all end the run.
A box with a person inside does not race: a contestant never waits for an answer.
If every contestant fails, the race fails with the last failure. Inside a `for each`
chain, each item races on its own.

## 💾 Save and share

A system saves as a bundle: a folder named `name.mas/`, or the same files zipped into one
file named `name.mas`. Here the system of "Conditions and loops" goes into `team.mas` and
comes back:

```python
mas.save("team.mas")  # save("team.mas", zipped=True) writes one zip file
team = MAS.load("team.mas")
print(team.describe()["id"] == mas.describe()["id"])  # True
```

`MAS.load` builds each model the bundle names on the endpoint of your configuration (see
Configuration). It finds each function where `callables.yaml` says it lives. Pass
`llm=`, `llms=` or `tools=[...]` to supply your own instead. The folder holds one file
per part:

```text
team.mas/
  xmas.yaml        # the wiring: inputs, outputs, nodes, edges, models by name
  callables.yaml   # where each function and tool lives, with typed reads and writes
  prompts/         # one plain-text file per prompt: writer.md, judge.md
  boxes/           # each box's system, as a bundle of its own
  examples/        # warm-up cases for verify(), optional
  optimize.yaml    # what an optimizer may tune, optional
  xmas.lock        # checksums of every file and of each function's source
```

`xmas.yaml` is the system's spec without its prompts. No prompt text sits in the wiring:

```yaml
version: 2
agents:
  - {name: writer, reads: [query, verdict], writes: {draft: text}}
  - {name: judge, reads: [query, draft, count_words], writes: {verdict: json}}
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
default_model: llama-3.1-8b
```

A prompt file is the text exactly as the model sees it. A short header may hold the
answer's schema, and nothing else; xmas strips it before the model sees the text.
`prompts/judge.md`:

```markdown
---
schema:
  fits: {type: bool, means: the draft answers the question in under 50 words}
  action: {type: text, one of: [publish, shorten]}
---
Question: {query}
Draft, {count_words} words: {draft}
Judge the draft.
```

`callables.yaml` locates code and never inlines it. A remote tool names its server, here
`web`, and the server's address sits in your configuration under `[tools.web]`:

```yaml
count_words:
  kind: python
  import: mytools.text:count_words
  reads: {draft: text}
  writes: {count_words: json}
web_search:
  kind: mcp
  server: web
  writes: {web_search: text}
```

A python entry may add `params:`, constant arguments bound when the bundle loads. A model
name may stand for another catalogue id and carry parameters, under `models` in
`xmas.yaml`:

```yaml
models:
  editor: {model: llama-3.1-70b, max_tokens: 2000, timeout_s: 120, temperature: 0.2}
```

The parameters are `max_tokens`, `timeout_s`, `temperature`, `top_p`, `seed` and
`reasoning_effort`. They apply to the models `MAS.load` builds.

`xmas.lock` records the checksum of every other file and of each function's source, for
`verify()` to compare. Saving again rewrites only the files whose content changed. A new
prompt from an optimizer changes one prompt file and the lock, and your comments in the
other files stay. A bundle never holds an endpoint, a key, a tool server's address, a
budget or a run. A bundle may name any code, so by default `MAS.load` imports nothing: a
function must be a `@tool` in a module your program has already imported, or be passed
in `tools=[...]`. `MAS.load(path, trust=True)` imports the modules `callables.yaml` names
and wraps plain functions; use it only for bundles you trust.

A version 1 system, such as the quickstart's, also saves as one YAML file:

```python
team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)  # the quickstart's system
team.to_yaml("team.yaml")
team = MAS.from_yaml("team.yaml", llm=llm)
```

Any system's spec is plain data. `to_dict()` gives it, and `MAS.from_dict` rebuilds the
system with the same models and functions:

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

A whole spec in a YAML file loads with `MAS.from_yaml(path, llm=llm, tools=[...])`.
`to_yaml` refuses a version 2 system and points to `save`. Every version 1 file loads,
runs and saves as before.

## Verify

`verify()` checks a system on three levels, each including the ones before it:

- **0, checksums:** every file and each function's source is unchanged since it was
  saved, and the system in memory matches the files.
- **1, dry:** the wiring check, every prompt rendered, every answer form compiled, every
  examples file read, and every model name looked up in the endpoint's catalogue. No
  model is called. This is the default level.
- **2, warm-up:** the examples run under a small budget. A step must write exactly what
  was recorded, and an agent's answer must pass its checks.

`save_examples` records one case per node from a run. Here the loaded team runs once and
keeps cases for the count and the judge, then runs them:

```python
trace = team.run("Why is the sky blue?")
team.save_examples(trace, nodes=["count", "judge"])
print(team.verify(2))
```

```text
level 0: unchanged since saved (bundle 9c7e2c30958e1c1d, 5 files)
level 1: no problems; model names checked against the endpoint catalogue
level 2: count  pass  0.0000 USD  <0.1 s
level 2: judge  pass  0.0100 USD  0.9 s
ok: 2 examples passed, 0.0100 USD
```

The cases land in `examples/warmup.yaml`, and the lock is refreshed. You can also write
cases by hand. A node case gives the values the node reads; a run case gives the system's
inputs and checks per output or node under `expect`:

```yaml
version: 1
budget: {cost_usd: 0.05, per example: {cost_usd: 0.03, wall_s: 120}}
examples:
- name: judge
  node: judge
  inputs: {query: "Why is the sky blue?", draft: Sunlight scatters off air., count_words: 4}
  recorded: {fits: true, action: publish}
  must: [{form: valid}, {path: verdict.fits, equals: true}]
- name: short-answer
  inputs: {query: "Why is the sky blue?"}
  expect: {judge: [{path: verdict.action, equals: publish}]}
```

A check is one of `form: valid`, `equals`, `contains`, `not contains`, `matches`, `line`,
`len`, `not empty`, `exists`, or `near` with `within`. Its `path` picks a value and a
field, such as `verdict.fits` or `pages[0]`. A picture input is a reference to a file
under `examples/media/`.

Without a budget, level 2 spends at most 0.05 USD in all and 0.02 USD per example. The
file's `budget` or `verify(2, budget=0.10)` sets another. A model whose cost is unknown
and that has no price fails its example, since no cap could bind. A report that is not
ok gives one `VerifyWarning`; `verify(strict=True)` raises `VerifyFailed` with the report
instead. `MAS.load(path, verify=1)` verifies while it loads, and `verify(2, only=["judge"])`
runs only that node's cases. Run it after a model switch, after an optimizer run and
before a release.

## The command line

The `xmas` command works on bundles from the shell:

```bash
xmas init team.mas         # a one-agent bundle to edit; --agents a,b for a chain
xmas check team.mas        # load and check, no model call
xmas show team.mas         # nodes and edges as text; --json for describe()
xmas run team.mas --query "Why is the sky blue?" --budget 0.10
xmas diff team.mas copy.mas  # file by file; with one bundle, against its lock
xmas verify team.mas --level 2
```

`run` takes other inputs as `--input name=value`: JSON for a json value, a path for a
picture. It prints the output, then one line with the status, the model calls, the cost
and the time. It refuses a model name the catalogue does not hold. `show`, `diff` and
`verify` take `--json`, and `verify --strict` fails when the report is not ok. The exit
status is 0 when all is well, 1 for a problem, and 2 for a usage error or a missing
bundle. `check`, `show`, `run` and `verify` take `--trust` to import the functions a bundle
names, from the current folder too.

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
`to_dict()` instead of a YAML file. `EvolutionaryTopologyOptimizer` searches the wiring.
It keeps steps, boxes, human nodes, frozen nodes, conditional edges and every `for each`
chain as they are, and drops any candidate the check refuses. Steps are never part of the
prompts an optimizer sees, and agents that share a prompt are optimized as one prompt.

## 🔗 xmas and promptolution

xmas builds on [promptolution](https://github.com/gepromptet/promptolution): a system is a
promptolution predictor and a task is a promptolution task, so promptolution's optimizers
run on any agent unchanged.

A one-agent system encodes to its bare prompt, with or without steps. So promptolution's
optimizers send the same requests to it as to a single prompt.

## Where this is going

> **Planned, not in the code yet.** Nothing in this section runs today.

### Learning from human picks

A preference task learns from the picks people make in `judge` nodes, with credit
per node, so an optimizer can improve a system against what people chose.

## 📚 Scientific Publications

Methods implemented in xmas:

- **CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — Zehle, 2026. [arXiv](https://arxiv.org/abs/2605.13295) · `CantanteOptimizer`, `ContrastiveAttributor`

xmas builds on:

- **promptolution: A Unified, Modular Framework for Prompt Optimization** — Zehle et al., 2026. [EACL 2026](https://aclanthology.org/2026.eacl-demo.21/)
- **CAPO: Cost-Aware Prompt Optimization** — Zehle et al., 2025. [AutoML 2025](https://proceedings.mlr.press/v293/zehle25a.html)

## 🏗 Components

- **`Agent`** – One node: a prompt template, the values it reads and the value it writes.
- **`Step`** – A node that runs a function or an MCP tool from the system's `tools=`, with no model call.
- **`Edge`** – One edge with its operator, an `EdgeKind`: plain, `when`, `else`, `while`, `switch`, `on error` or `first of`.
- **`Box`** – A whole system used as one node of another.
- **`Human`** – A node that asks a person, in one of four forms: `ask`, `judge`, `approve` or `edit`.
- **`MAS`** – The system and the predictor: agents, steps, edges, models and tools; runs a query and keeps its trace.
- **`Trace`** – The per-agent record of one run: rendered prompts, outputs, tokens, cost, time, tool calls, status.
- **`ForEach`** – The `for each` form of a chain: the item, the list it comes from, and options such as `at_once` and `key`.
- **`Ref`** – A picture, a sound, a document, a table or a file as a value: where it is, its type, size and checksum.
- **`Capabilities`** – What a model accepts and produces beyond text, such as `Capabilities(accepts=("image",))`.
- **`Event`** – One start or end of a visit or an item, a pause, a resume or a stop, passed to `on_event` as the run goes.
- **`RunStore`** – Where paused runs, their questions and answers are kept: in memory, or in one SQLite file.
- **`Budget`** – Limits on calls, cost and wall time for each run, and on model calls in flight.
- **`Config`** – Where models and tools are served: endpoint, key, catalogue, prices and tool servers.
- **`VerifyReport`** – What `verify()` found on each level, with each example's result, cost and time.
- **`Health`** – Signals read from a finished run: stuck loops, empty answers, retries, caps hit.
- **`MASTask`** – A dataset plus a reward over the system's answers, e.g. `GSM8KTask`.
- **`APILLM`** – Any OpenAI-compatible endpoint, with scoring, tool turns and a per-call log.
- **`MASOptimizer`** – Propose, evaluate through the task, select; CANTANTE and topology search build on it.

## Configuration

A bundle names its models and tool servers. Where they are served lives in your
configuration: the environment, then a TOML file, `$XMAS_CONFIG` or else
`~/.config/xmas/config.toml` (under `$XDG_CONFIG_HOME` when it is set). A missing file is
no error. What you pass in code wins over both, such as
`MAS.load(path, config=Config.load(base_url=...))`. A file with every table:

```toml
[endpoint]
base_url = "http://localhost:8000/v1"
api_key_env = "GATEWAY_KEY"        # or api_key = "..."
catalogue = ["llama-3.1-8b"]       # optional; else GET {base_url}/models
stream = false

[models."llama-3.1-8b"]
price = {input_per_m = 0.6, output_per_m = 2.5}
stream = true

[tools.web]
url = "https://tools.example.com/mcp"
key_env = "WEB_TOOLS_KEY"          # or key = "..."

[store]
media = "~/.cache/xmas/store"
runs = "~/.cache/xmas/runs.sqlite"
```

Prices and streaming hold per model id, for the models `MAS.load` builds. A tool server
is named by `server:` in `callables.yaml` and found under `[tools.<name>]`. The
environment variables:

| Variable | Meaning |
|---|---|
| `XMAS_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `XMAS_API_KEY` | API key for that endpoint |
| `XMAS_MODELS` | Model ids, comma separated, that stand for the catalogue, so names check offline |
| `XMAS_CONFIG` | The configuration file, instead of `~/.config/xmas/config.toml` |
| `XMAS_TOOLS_KEY` | Key for a tool server whose entry has none |
| `XMAS_STORE` | Folder for stored pictures and other media; default `~/.cache/xmas/store` |
| `XMAS_RUNS` | File where paused runs are kept, so they resume after a restart; default in memory |

The catalogue is the list of model ids the endpoint serves. It is `catalogue` or
`XMAS_MODELS` when set, else the endpoint's answer to `GET /models`, kept for five
minutes per endpoint and key. Offline, a kept answer of any age stands in. A model name
the catalogue does not hold gives a `ModelNameWarning` with the closest ids, such as
`llama-3.1-8b` for `llama-3-1-8b`. xmas never swaps in another name: the system refuses
to run until the name is fixed, and `verify(1)` checks again. Without an endpoint or a
catalogue, names go unchecked.

No key appears in a bundle, a record, a `repr` or the kept catalogue. A configuration
file that holds a key and that others can read gives a warning.

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
