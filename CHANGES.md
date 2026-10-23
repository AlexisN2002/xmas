# Changes in xmas, newest first

## Phase 1: typed errors, budgets and richer records (23 Oct 2026)

What a user of xmas can do now that they could not before:

- Give a system a budget: `MAS(..., budget=Budget(max_cost_usd=2.0, max_wall_s=600, max_calls=40))`. A call starts only while the run is below every limit, so a budget of 0 spends nothing. When a limit is reached the run ends with status `over_budget` instead of an exception, and the message names the limit, the amount and the call it refused.
- Read what every agent call cost: each record now carries latency, cost in USD, the provider's token counts (cached and reasoning tokens included where the provider reports them), the finish reason, the error kind and the model that actually served the call, next to the binding name it had before.
- Keep the work of a failed run: when an agent fails, the trace returns every record that finished before it, a run status (`ok`, `failed`, `over_budget`) and a typed error kind (money, rate limited, outage, timeout, length cut, bad format, refusal, budget, unknown). The error text keeps its old form.
- Pay each item once: when one item of a batch fails, the other items keep their results and are not run again.
- Cost is never guessed as zero. It comes from the response when the gateway reports it, else from a price per model you pass to the client, else it is unknown and a cost budget refuses the next call until a price is known.

What changed in the format: nothing in the spec file. Version 1 files load and run as before. Trace and record dictionaries keep their old keys in the old order and add a key only when it is set.

What stays the same: the request a one-agent system sends (checked byte for byte), the bare-prompt encoding of a single agent, promptolution's optimizers, and the public calls `run`, `run_batch`, `predict`, `copy`, `to_yaml` and `from_yaml`. One difference on xmas's own model class: a permanent provider error (bad request, key or money refused) is sent once instead of six times; the old behaviour is one flag away (`full_response=False`).
