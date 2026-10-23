# API reference

Generated from docstrings. Everything on this page is public surface; anything
not on it is an implementation detail and may change without notice.

## Defining a system

::: xmas.agent.Agent

::: xmas.mas.MAS

## Topology

The graph is a plain `networkx.DiGraph` over agent names plus two sentinels, so a
topology optimizer is an ordinary graph-rewriting component.

A cycle is expressed as an ordinary edge. The engine classifies the edge that
closes it as a back edge and compiles it to a conditional branch, so the agent
that closes the loop decides whether to take it by emitting `<continue>` in its
response. Two guards bound the loop regardless of what a prompt says: that
agent's own visit count, and `max_steps` behind it.

::: xmas.graph
    options:
      members:
        - START
        - END
        - build_graph
        - validate_graph
        - agent_names
        - back_edges
        - forward_predecessors
        - join_nodes
        - terminal_agent

## Traces

The record a run leaves behind, and the input an attributor computes on.

::: xmas.trace.Trace

::: xmas.trace.AgentRecord

::: xmas.trace.ToolCall

::: xmas.trace.TokenUsage

## Budgets and errors

`MAS(..., budget=Budget(...))` limits every run, and every item of a batch, on
its own. A call starts only while the run is below every limit, so a limit of
`0` spends nothing, and calls already in flight may overshoot a budget, by at
most one model call, with its retries, per parallel branch. A cost limit stops
the run before the next call when a call reported no cost and its model has no
price, because a limit that cannot be checked would not bind. A run stopped
this way is data, not an exception: its trace has status `over_budget`, and
every failed trace carries an `error_kind`.

On xmas's own model class, `APILLM`, permanent errors (a 4xx other than 408,
409 and 429, or a 4xx that names money) are not retried;
`APILLM(full_response=False)` restores promptolution's path for plain agents,
which retries every error.

::: xmas.budget.Budget

::: xmas.errors
    options:
      members:
        - ErrorKind
        - RunStatus
        - XmasError
        - BudgetExceeded
        - Stopped
        - EmptyResponse
        - BadFormat
        - classify
        - kind_of_status
        - kind_of_finish
        - status_for

## Models

Each backend subclasses its promptolution counterpart and adds `score()`, the
teacher-forced sibling of generation.

`APILLM(full_response=None)`, the default, sends plain agents' calls through
xmas's own path, which sends the same request and keeps the provider's tokens,
cost, finish reason and served model on the agent's record, as tool turns
already do; a subclass that overrides a hook of promptolution's path keeps that
path, and `full_response=False` keeps it for any model. `APILLM(price=TokenPrice(...))` prices the tokens of a call
whose response reports no cost; without a price, that cost stays unknown
(`None`), never `0`.

::: xmas.llms.api_llm.APILLM

Per-call records: pass `call_log=CallLog()` and every call leaves a `CallRecord` with a
`call_id` (UUID4), a `ts` (start time, ISO 8601 with offset), the `params` it was sent
(`max_tokens` plus `call_kwargs`, minus `extra_headers`) and the provider's `response_id` (`response_ids` for a batch).
`CallLog(capture_text=True)` also records `prompt`, `system_prompt` and `response`; a batch
stays one record and fills `prompts`, `system_prompts`, `responses` instead.
A provider or tool error raises `XmasError` (under Budgets and errors), which
carries the HTTP `status` when there is one.

::: xmas.llms.scoring.ScoredSequence

::: xmas.llms.meta.TokenPrice

::: xmas.llms.meta.CallMeta

## Tools

Functions an agent may call while it answers: `Agent.tools` names resolve
against `MAS(..., tools=[...])` when the system is built. `MCPTools` needs the
`mcp` extra. A tool turn sends sampling parameters (`temperature`, `top_p`,
`stop`, ...) only when they are set in the model's `call_kwargs`, never as
defaults. Every requested call runs, since a tool call need not be idempotent.

::: xmas.tools
    options:
      members:
        - Tool
        - tool
        - ToolSource
        - ToolCallRequest
        - LocalTools
        - run_step

::: xmas.tools.mcp.MCPTools

::: xmas.llms.masking
    options:
      members:
        - assistant_span
        - mask_logprobs

## Version 2: steps, typed values, answer forms

A system is version 2 when it uses anything below (or `MAS(..., version=2)`);
a system that uses none of it is version 1, with its own spec format. What a
version 2 system can do:

- **Steps.** `MAS(..., steps=[Step("count", uses="count_words")])` runs a
  callable from `tools=` as a node, with no model call. The same callable can
  also be an agent's tool. A step reads its parameters and writes one value
  named like the callable; `Step(reads={param: value}, writes={write: value})`
  renames them, and `tool(writes={"counts": "json"})` declares other writes.
- **Typed values.** Every value is `text` or `json` (`TEXT`, `JSON`); a
  callable's `string` parameters read text, the others json, and its write is
  text when the function returns `str` (or the tool is remote), else json. A
  json value reaches a prompt as one line of JSON. Images, files and lists of
  them are type words the format knows, but only text and json run: the check
  refuses the others.
- **Answer forms.** `Agent(..., output_type="json", schema={...})` makes the
  agent answer in JSON. Its prompt ends with format lines no optimizer can
  drop; an answer that does not fit gets exactly one retry, then the run fails
  with `BadFormat` (kind `bad_format`). The record keeps the rejected answer.
- **Merges and joins.** A node fed by several edges is a merge when its edges
  never arrive in one pass (it runs on whichever arrives) or a join when it
  waits for one edge of every group (it runs once per pass); the check refuses
  any other node and says which edges clash.
- **Checked when built.** `check_system` lists every problem at once: a read
  nobody writes before the node runs, two writers of one value (unless
  `state={value: {"merge": "last"}}`), a placeholder that is not a read, a type
  that does not fit a parameter, a reserved name (the type words such as
  `text` and `json`, and `visits`, `cost`, `len`, ...), an unknown callable.

The answer-form dialect, one entry per field:
`{type, means?, optional?, one of?, fields?}`, where `type` is `text`, `bool`,
`int`, `number`, `json`, `object` or `list of <type>`, and `fields` describes
an object or a list's objects.

A version 2 spec (`version: 2`) has the keys `inputs`, `agents` (`reads`,
`writes: {name: type}`, `schema`, `tools`, `model`), `steps`, `boxes`, `edges`
(a string is a chain, `a -> b -> c`), `state`, `frozen`, `health`, `callables`
(what each callable reads and writes), `default_model` and `prompts` (one per
agent). Models
appear by name only and keys never appear: endpoints and keys come from
configuration, never from the spec.

Refused, with a message saying so: `to_yaml` on a version 2 system (its file
format is the `.mas` bundle; `to_dict()` gives its spec), and `run(query)` on a
system whose inputs are not exactly `{query: text}`.

::: xmas.nodes.Step

::: xmas.edges
    options:
      members:
        - Edge
        - EdgeKind

::: xmas.values
    options:
      members:
        - ValueType
        - TEXT
        - JSON
        - parse_type
        - LIST_PREFIX

::: xmas.schemas
    options:
      members:
        - compile_form
        - AnswerForm
        - FormError
        - RETRY_NOTE

::: xmas.check
    options:
      members:
        - check_system
        - SystemInfo

::: xmas.routes
    options:
      members:
        - Join

## Version 2: operators, boxes, health

Edges decide where a run goes next. In a spec, a chain `a -> b` is plain (always
taken); a one-key mapping gives the operator:

- **`when`.** `judge -> publish: {when: verdict.fits}` is taken when the
  predicate holds, judged on the node's fresh writes.
- **`else`.** `judge -> apology: else` is taken when no `when`, `while` or
  `switch` edge of the same node is.
- **`while`.** `judge -> writer: {when: not verdict.fits, max_visits: 3}` loops
  back until its target ran `max_visits` times; then the record is `capped` and
  the `else` edge is taken, even when a later `when` edge holds.
- **`switch`.** `judge -> shorten: {switch: verdict.action}` is taken when the
  path equals the case (the target's name unless `case:` says otherwise); a
  value outside the cases takes `else`.
- **`on error`.** `count -> apology: {on error: [timeout, outage]}` (or `any`)
  catches the node's failure of those kinds: the visit keeps a failed record and
  the run goes on. Budget and stop are never caught.

Predicates are a small language, never `eval`: paths (`verdict.fits`),
literals, `== != < <= > >= in`, `and or not`, `exists(path)`, `len(x)`,
`visits`, `cost`, `elapsed`, and registered callables that return a bool. The
check refuses an unknown name before a run.

A **box** (`Box(name, system, reads, writes)`, spec `boxes:`) runs a whole
version 2 system as one node: `reads` maps its inputs from outer values,
`writes` its values back out. It spends from the outer budget, and its record
holds the inner trace (`Trace.walk()` yields every record with its box path).

::: xmas.nodes.Box

`frozen: [judge]` (`MAS(frozen=...)`) keeps a node as it is through every
optimizer: no prompt, model, read, write or edge of it changes, none removes it.
The topology search also leaves conditional edges and their endpoints alone,
keeps steps and boxes, and drops a candidate the check refuses.

Every run gets **health** signals in `Trace.health`, for information only:
`retried`, `bad_format`, `error_routed`, `empty`, `stuck`, `unfilled`, `leak`,
`long` and `short` per record, how often each `while` edge was taken, and each
set budget limit's use. A spec tunes them with `health: {length_factor,
min_samples, leak}`; a healthy run's trace dict has no `health` key.

`MAS.describe()` returns plain data: every node with its kind, model name,
reads, writes and `frozen`; every edge with its kind and condition; the join of
each node fed by several edges; an `id` hashed from it. Like the spec, it names
models only: endpoints and keys come from configuration.

::: xmas.predicates
    options:
      members:
        - parse_predicate
        - Predicate
        - PredicateError

::: xmas.health
    options:
      members:
        - Health
        - HealthSettings
        - Flag
        - summarize

## Prompts

::: xmas.prompts
    options:
      members:
        - safe_format
        - encode_prompts
        - decode_prompts

## Tasks

::: xmas.tasks.base.MASTask

::: xmas.tasks.gsm8k.GSM8KTask

::: xmas.tasks.gsm8k.extract_number

## Optimization

::: xmas.optimization.optimizer.MASOptimizer

::: xmas.optimization.attribution.Attributor

::: xmas.optimization.attribution.UniformAttributor

::: xmas.optimization.attribution.ContrastiveAttributor

::: xmas.optimization.updaters.PromptUpdater

::: xmas.optimization.updaters.AgentPromptView

::: xmas.optimization.prefix_search.PrefixSearchOptimizer

### CANTANTE

Attribution-guided prompt optimization: every agent's prompt is improved at once,
by its own promptolution optimizer, against a single shared system evaluation
that an attributor splits back into per-agent credit.

::: xmas.optimization.cantante.CantanteOptimizer

::: xmas.optimization.broker
    options:
      members:
        - Board
        - ProxyTask
        - Request
        - rounds

### Topology search

The other half of the decomposition: the graph is the component, and the same
loop improves it. Operators work on the agent-level edge set, with the sentinel
wiring left to `build_graph`, so a candidate is legal exactly when a `MAS` can be
built from it. On a version 2 system they move only what `Fixed` leaves free,
and the system's own check decides which candidates are returned.

::: xmas.optimization.evolution.EvolutionaryTopologyOptimizer

::: xmas.optimization.graph_updater.GraphUpdater

::: xmas.optimization.topology
    options:
      members:
        - TopologyBudget
        - Fixed
        - mutate
        - crossover
        - repair
        - mas_edges
        - is_valid
        - within_budget
        - sources
        - terminals
        - cycles

### Prompt synthesis

Topology search moves information around; something has to write the sentence
that uses it.

::: xmas.optimization.synthesis.LLMPromptSynthesizer

::: xmas.optimization.synthesis.MechanicalSynthesizer

::: xmas.optimization.synthesis.Rewrite

## Serialization

::: xmas.spec
    options:
      members:
        - validate_spec
        - validate_spec_v2
