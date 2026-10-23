# Changes in xmas, newest first

## Phase 2b: conditions, loops, error routes, boxes and health (23 Oct 2026)

What a user of xmas can do now that they could not before:

- Let a node decide where the run goes: `when` (taken if a condition on its fresh answer holds), `else`, and `switch` (pick the edge whose case equals a field, for example `verdict.action`). Conditions are a small checked language, never code: fields, comparisons, `and or not`, `exists`, `len`, `visits`, `cost`, `elapsed`.
- Loop with a cap: a `while` edge goes back until its target ran `max_visits` times, then the record is marked capped and the `else` edge takes over.
- Route a failure by its kind: `on error: [timeout, outage]` (or `any`) sends the run to a fallback node and keeps the failed record, with the cost of every call the visit paid; a budget stop is never caught. A failed version 2 run keeps its failed record too. A node whose only edges are `on error` edges ends the run when it succeeds.
- Put a whole system inside another as one node (a box). It spends from the outer budget, and the trace keeps its records nested, with `Trace.walk()` naming the path.
- Mark nodes `frozen`: no optimizer changes their prompt, model, wiring or edges. Topology search now works on version 2 systems and keeps steps, boxes, frozen nodes and conditions as they are.
- Read health signals on every trace: retried, bad format, routed errors, empty, stuck, unfilled placeholders, leaked markup, unusually long or short answers, how often each loop ran and whether it hit its cap, and how much of each budget limit was used. For information only; a healthy run shows none.
- A node fed by several edges is a merge (runs on whichever arrives) or a join (waits for one of each group, once per pass); the check refuses anything else and names the clashing edges.

What changed in the format: edges in a version 2 spec may carry `{when: ...}`, `{when: ..., max_visits: n}`, `{switch: ...}`, `else` or `{on error: [...]}`; new sections `boxes`, `frozen` and `health` (length factor, minimum samples, extra leak patterns).

What stays the same: every version 1 system, its requests and optimizers; a version 1 trace gains a `health` key only when a signal fires, so a healthy run's trace has none. Every tool call runs, as before; answering a repeated call from its earlier result is opt-in (`xmas.tools.DEDUPE_TOOL_CALLS = True`). One version 1 difference: a tool agent whose last turn is cut by `max_tokens` or refused before it wrote anything now fails with `length_cut` or `refusal`, as a plain agent always did, instead of answering `""`. Still to come: saving a version 2 system as a `.mas` bundle.

## Phase 2a: steps, typed values and answer forms (23 Oct 2026)

What a user of xmas can do now that they could not before:

- Run plain code as a node: `MAS(..., steps=[Step("count", uses="count_words")])` runs a function (or an MCP tool) between agents without a model call. The same function can also be a tool an agent calls.
- Pass structured data: every value is text or JSON. A JSON value reaches a prompt as one line of JSON, and a function gets the real object, not a string.
- Ask an agent for JSON with a fixed shape: `Agent(..., output_type="json", schema={"fits": {"type": "bool"}})`. The prompt ends with format lines no optimizer can drop; an answer that does not fit gets exactly one retry, then the run stops with a `bad_format` error. On the recorded ELLIOT answers, 461 of 466 checker answers and all 89 editor answers pass the first time.
- Find every wiring mistake when the system is built, listed together: a value read before anything writes it, two writers of one value, a placeholder the agent does not read, a JSON value sent where a function wants text, an unknown function.
- A node fed by several others waits for all of them and runs once.

What changed in the format: a new spec version 2 (`version: 2`) with `inputs`, `steps`, typed `reads` and `writes`, `schema` on agents, a `callables` list and a `prompts` section. Models appear by name only, and keys never appear. Version 1 files load, run and save exactly as before.

What stays the same: every version 1 system, its requests, traces and optimizers. Not yet in 2a (they come in 2b): conditions, loops, boxes, `frozen`, health signals and topology search on one. Saving a version 2 system comes with the `.mas` bundle; `to_yaml` refuses one until then.

## Phase 1: typed errors, budgets and richer records (23 Oct 2026)

What a user of xmas can do now that they could not before:

- Give a system a budget: `MAS(..., budget=Budget(max_cost_usd=2.0, max_wall_s=600, max_calls=40))`. A call starts only while the run is below every limit, so a budget of 0 spends nothing. When a limit is reached the run ends with status `over_budget` instead of an exception, and the message names the limit, the amount and the call it refused.
- Read what every agent call cost: each record now carries latency, cost in USD, the provider's token counts (cached and reasoning tokens included where the provider reports them), the finish reason, the error kind and the model that actually served the call, next to the binding name it had before.
- Keep the work of a failed run: when an agent fails, the trace returns every record that finished before it, a run status (`ok`, `failed`, `over_budget`) and a typed error kind (money, rate limited, outage, timeout, length cut, bad format, refusal, budget, unknown). The error text keeps its old form: a failed plain call reads `RuntimeError: LLM batch failed: ...` whether or not the model reports call metadata (a version 2 failure reads `Type: message`).
- Pay each item once: when one item of a batch fails, the other items keep their results and are not run again.
- Cost is never guessed as zero. It comes from the response when the gateway reports it, else from a price per model you pass to the client, else it is unknown and a cost budget refuses the next call until a price is known.

What changed in the format: nothing in the spec file. Version 1 files load and run as before. Trace and record dictionaries keep their old keys in the old order and add a key only when it is set.

What stays the same: the request a one-agent system sends (checked byte for byte), the bare-prompt encoding of a single agent, promptolution's optimizers, and the public calls `run`, `run_batch`, `predict`, `copy`, `to_yaml` and `from_yaml`. One difference on xmas's own model class: a permanent provider error (bad request, key or money refused) is sent once instead of six times; the old behaviour is one flag away (`full_response=False`) for plain agents; a tool turn sends a permanent error once either way.
