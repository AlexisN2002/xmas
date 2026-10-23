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

::: xmas.llms.api_llm.ScoringUnsupportedError

::: xmas.llms.scoring.ScoredSequence

::: xmas.llms.scoring.SupportsScore

::: xmas.llms.meta.TokenPrice

::: xmas.llms.meta.CallMeta

## Tools

Functions an agent may call while it answers: `Agent.tools` names resolve
against `MAS(..., tools=[...])` when the system is built. `MCPTools` needs the
`mcp` extra. A tool turn sends sampling parameters (`temperature`, `top_p`,
`stop`, ...) only when they are set in the model's `call_kwargs`, never as
defaults.

::: xmas.tools
    options:
      members:
        - Tool
        - tool
        - ToolSource
        - ToolError
        - ToolAccessError
        - ToolCallRequest
        - SupportsToolCalls
        - LocalTools

::: xmas.tools.mcp.MCPTools

::: xmas.llms.masking
    options:
      members:
        - assistant_span
        - mask_logprobs

## Version 2: steps, typed values, answer forms

A system is version 2 when it uses anything below (or `MAS(..., version=2)`);
every version 1 system runs, serializes and compiles exactly as before. What a
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
  them are words the format already knows; they run in phase 3.
- **Answer forms.** `Agent(..., output_type="json", schema={...})` makes the
  agent answer in JSON. Its prompt ends with format lines no optimizer can
  drop; an answer that does not fit gets exactly one retry, then the run fails
  with `BadFormat` (kind `bad_format`). The record keeps the rejected answer.
- **Joins.** A node fed by several edges waits for all of them and runs once.
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
`writes: {name: type}`, `schema`, `tools`, `model`), `steps`, `edges` (a
string is a chain, `a -> b -> c`), `state`, `callables` (what each callable
reads and writes), `default_model` and `prompts` (one per agent). Models
appear by name only and keys never appear; in the `.mas` bundle (phase 5)
endpoints and keys come from configuration, never from the bundle.

Refused until phase 2b, with a message saying so: conditional edges (`when`,
`else`, `while`, `switch`, `on error`) and every cycle, boxes, `frozen`,
`health`, `to_yaml` on a version 2 system, and the topology updater on one.
`run(query)` needs the inputs to be exactly `{query: text}`.

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

## Prompts

::: xmas.prompts
    options:
      members:
        - safe_format
        - extract_placeholders
        - encode_prompts
        - decode_prompts

## Tasks

::: xmas.tasks.base.MASTask

::: xmas.tasks.gsm8k.GSM8KTask

::: xmas.tasks.gsm8k.extract_number

## Probing

Generation-free evaluation. A probing task wraps any other task and scores candidates
by teacher-forcing a shared pool of references whose reward is already known, instead
of decoding a fresh answer per candidate. Every optimizer runs against it unchanged,
which is design principle P2.

The estimator is CALINKA's importance-sampling form: a candidate is credited for the
share of the *population's* probability mass it puts on references that scored well.

::: xmas.tasks.probing.ProbingTask

::: xmas.tasks.probing.Reference

::: xmas.tasks.probing.ReferencePool

::: xmas.tasks.probing.ProbeTensor

::: xmas.tasks.probing.Estimator

::: xmas.tasks.probing.ImportanceSampling

::: xmas.tasks.probing.AgreementReadout

::: xmas.tasks.probing
    options:
      members:
        - measure_agreement
        - negative_control
        - materialize
        - importance_weights
        - rank_agreement

## Optimization

::: xmas.optimization.base.MASOptimizer

::: xmas.optimization.attribution.Attributor

::: xmas.optimization.attribution.UniformAttributor

::: xmas.optimization.attribution.ContrastiveAttributor

::: xmas.optimization.attribution.UNKNOWN

::: xmas.optimization.updaters.Updater

::: xmas.optimization.updaters.TopologyUpdater

::: xmas.optimization.updaters.PromptUpdater

::: xmas.optimization.updaters.AgentPromptView

::: xmas.optimization.reference.PrefixSearchOptimizer

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
built from it.

::: xmas.optimization.evolution.EvolutionaryTopologyOptimizer

::: xmas.optimization.graph_updater.GraphUpdater

::: xmas.optimization.topology
    options:
      members:
        - TopologyBudget
        - mutate
        - crossover
        - repair
        - mas_edges
        - is_valid
        - within_budget
        - agent_edges
        - sources
        - terminals
        - cycles

### Prompt synthesis

Topology search moves information around; something has to write the sentence
that uses it.

::: xmas.optimization.synthesis.PromptSynthesizer

::: xmas.optimization.synthesis.LLMPromptSynthesizer

::: xmas.optimization.synthesis.MechanicalSynthesizer

::: xmas.optimization.synthesis.Rewrite

## Serialization

::: xmas.spec
    options:
      members:
        - MASSpec
        - AgentSpec
        - validate_spec
        - SPEC_VERSION
        - MASSpecV2
        - validate_spec_v2
        - SPEC_VERSION_2
