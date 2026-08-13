# Careem Care Agent

A proactive, state-driven Care agent that opens an outbound text conversation with a merchant
when their order is delayed, negotiates strictly within an external policy, and hands off to a
human when the policy says it must.

Everything below runs **offline with no API key** — the state machine, policy engine, tool
chain, guardrails, and the full 30-run evaluation are all deterministic and reproducible.

---

## The core invariant

**The LLM never makes a decision.** A deterministic policy engine, reading `policy/policy.json`,
originates every action. The model sits at the two edges only:

- **inbound** — a classifier turns free text into five fixed booleans (structured output, so it
  has no channel to emit an action);
- **outbound** — a generator phrases an *already-decided* action, and a guardrail verifies the
  draft before it can reach the merchant.

Neither edge can originate an action or a promise. A prompt injection can at most flip a
boolean, which routes into a policy branch that is safe by construction.

```
merchant text ──▶ classifier ──▶ Event ──▶ mailbox ──▶ reducer ──▶ policy engine
                                                                        │
                                      effects ◀─────────────────────────┘
                                         │
          ┌──────────────────────────────┼───────────────────────────────┐
          ▼                              ▼                               ▼
  generator + guardrail            tool broker                    escalation
    (phrasing only)          (chaining, idempotency)         (ticket + handoff)
```

All inputs for one order — merchant messages, backend events, tool results, timers — go onto
**one ordered mailbox keyed by `order_id`** and are applied one at a time. That single-writer
property is what makes the concurrency story tractable; see [PLAN.md](PLAN.md) §1.

---

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then, in order of what they prove:

```bash
python scripts/verify_policy.py    # the policy pack is valid and safe
python -m pytest -q                # 279 tests
python -m care_agent.cli           # watch one session play out
python run_all.py                  # the full evaluation -> report.json + report.md
```

### With Docker

```bash
docker build -t care-agent .
docker run --rm -v "$PWD":/out care-agent \
    python run_all.py --json /out/report.json --md /out/report.md
```

### Against the real Claude API

Set `ANTHROPIC_API_KEY`, then add `--live` to either entry point. Models used:
Haiku 4.5 (classifier), Sonnet 5 (generator and adversarial personas), Opus 5 (judge — a
different tier from the agent, so it is not grading its own prose).

---

## Running a session

```bash
python -m care_agent.cli                        # built-in demo scenario
python -m care_agent.cli --scenario my.json     # a scripted scenario
python -m care_agent.cli --live                 # real Claude API
```

The report shows the conversation, the policy action and rule behind each message, the FSM
trajectory, tool calls, guardrail blocks, and any escalation ticket. Offline, replies come from
the pre-approved per-language templates (labelled `fallback`) — honest, since there is no
phrasing model available.

## Running the evaluation

```bash
python run_all.py           # 10 scenarios x 3 adversarial personas = 30 runs
```

Writes `report.json` (one entry per run, in the schema the assessment specifies) and
`report.md` (human summary). Exits non-zero if any run fails, so it can gate CI.

**Objective metrics are decided by the policy engine, not by a model.** `trajectory_correctness`
and `policy_compliance` are exact checks — the scorer re-derives what *should* have happened by
running the pure engine over each scenario's own context, so changing a threshold in
`policy.json` changes the expectation automatically. The LLM judge only grades what has no exact
answer (promise semantics, language match, coherence) and is optional.

`guardrail_violations` counts unauthorized promises that **reached the merchant** — guardrail
*failures*. Drafts the guardrail stopped are *successes*, reported separately as
`guardrail_blocks`.

---

## The policy pack

Every threshold, tier, cap, rule ordering, and override lives in `policy/policy.json`. No
prompt or code path contains a threshold, rule ID, tier, or amount — two tests assert this by
grepping the system prompts.

```bash
python scripts/verify_policy.py
```

Validates the file's shape against a JSON Schema **and** six safety invariants a schema cannot
express: delay coverage `0..inf` with no gaps or overlaps, first-hit ordering with disjoint
tiers, **R6 living in preemption rather than the rule list** (R1–R5 already cover 0..∞, so a
list-ordered R6 could never fire), an explicitly declared 40-minute boundary, override-map
integrity, and cap/precedence sanity.

The script also prints the **policy authoring gaps** — places the source policy is silent, where
the behaviour is an adopted decision rather than a stated rule. These are surfaced in the
evaluation report too, so they stay visible for human sign-off instead of being quietly patched.

---

## Layout

```
policy/              policy.json + policy.schema.json
scripts/             verify_policy.py
src/care_agent/
  domain/            typed models (session state, classifier flags, action envelope, events)
  policy/            loader + verification, and the pure decision pipeline
  core/              FSM reducer, single-writer mailbox, stores
  tools/             tool stubs, idempotency, the broker (chaining, retries, rollback)
  guardrails/        unauthorized-promise block, loop guard, language-invariant normalization
  llm/               classifier, generator, templates, client (+ offline stand-in)
  eval/              adversarial simulator, scenario runner, metrics, judge, report
  server/            optional MCP testing surface (see TESTING.md)
  agent.py           the orchestrator that assembles all of the above
  cli.py             run one session
eval/personas/       3 adversarial merchant suites
eval/scenarios/      10 scenarios
tests/               279 tests
run_all.py           the full evaluation suite
```

## Deliverables

| Assessment deliverable | Where |
|---|---|
| 1. Agent engine codebase | `src/care_agent/` — FSM (`core/`), orchestrator (`agent.py`), guardrails (`guardrails/`), tools (`tools/`) |
| 2. Policy config + verification script | `policy/policy.json`, `scripts/verify_policy.py` |
| 3. Adversarial evaluation framework | `eval/personas/`, `eval/scenarios/`, `src/care_agent/eval/` |
| 4. Pass/fail execution report | `report.json`, `report.md` (regenerate with `python run_all.py`) |
| 5. Architectural write-up | [WRITEUP.md](WRITEUP.md) |

## Driving it from your own agent (optional)

An MCP server exposes the agent as five tools, so a reviewer's own AI agent can drive it —
start a session, play the merchant, fire backend events, read the trace.

There is no web UI, because for this audience the chat window *is* the UI. Each turn renders as
a **decision card** naming the rule that decided it, with the structured JSON alongside it:

```
### escalate  <-  rule R6

> Merchant: عايز اكلم حد
Agent: سأحوّلك إلى زميل من فريقنا وسيتواصل معك قريبًا.

| Rule that decided | R6 - preemption, evaluated before the rule list |
| Guardrail         | PASS - clean                                   |
| FSM state         | AWAITING_MERCHANT_REPLY -> ESCALATED           |

Financial tools: none called - cancel_order and issue_merchant_credit are not
in the engine's action vocabulary at all
```

It also ships **six guided probes** as MCP prompts (refund extraction, prompt injection, R6 in
three languages, mid-conversation events, outage mode, and a full sweep), and exposes
`policy://policy.json` and `eval://report.json` as **resources** — so a reviewer can read the
rule table first, predict what should happen, and then check.

```bash
pip install -r requirements-server.txt
MCP_AUTH_TOKEN=my-secret python -m care_agent.server.mcp_server
```

See **[TESTING.md](TESTING.md)** for connection details and suggested probes. This layer is
entirely optional — nothing graded depends on it, and the core install does not pull in its
dependencies.

## Design documents

- **[PLAN.md](PLAN.md)** — architecture, the 14 locked policy-gap decisions, build plan.
- **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** — the open design forks, their trade-offs, and
  what was chosen, written to be readable without the assessment brief.
- **[DESIGN_AUDIT.md](DESIGN_AUDIT.md)** — a 78-finding adversarial audit of the spec's gaps and
  production risks, which fed the decisions above.

## Tests

```bash
python -m pytest -q
```

No test requires a network or an API key; the one live smoke test skips itself without
`ANTHROPIC_API_KEY`.
