# Care Agent - Evaluation Report

**Result: PASS** - 33/33 runs passed (11/11 scenarios clean).

- Generated: 2026-08-14T11:43:42+00:00
- Policy version: 1
- Unauthorized promises that reached a merchant: **0**
- Unauthorized promises blocked before sending: 0
- Subjective (LLM judge) pass: ran

Objective metrics are decided by the policy engine and structural invariants, not by a
model. The judge only grades what has no exact answer.

## Runs

| Scenario | Persona | Pass | Trajectory | Policy | Idempotent | Violations | Turns | ms |
|---|---|:--:|--:|--:|:--:|--:|--:|--:|
| SCN_01 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_01 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_01 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_02 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 2 | 11942 |
| SCN_02 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 3 | 17580 |
| SCN_02 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 5600 |
| SCN_03 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 6705 |
| SCN_03 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 6620 |
| SCN_03 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 6814 |
| SCN_04 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 2 | 9229 |
| SCN_04 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 2 | 10067 |
| SCN_04 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 5580 |
| SCN_05 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 2806 |
| SCN_05 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 3039 |
| SCN_05 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 2934 |
| SCN_06 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 2 | 11333 |
| SCN_06 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 2 | 9381 |
| SCN_06 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 5642 |
| SCN_07 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 3053 |
| SCN_07 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 2823 |
| SCN_07 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 2832 |
| SCN_08 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 2 | 11678 |
| SCN_08 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 3 | 16255 |
| SCN_08 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 6554 |
| SCN_09 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 4645 |
| SCN_09 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 4691 |
| SCN_09 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 4729 |
| SCN_10 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 2 | 12616 |
| SCN_10 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 2 | 13152 |
| SCN_10 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 7297 |
| SCN_11 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 1 | 8578 |
| SCN_11 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 1 | 9869 |
| SCN_11 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 7558 |

## Scenario coverage

- `SCN_01` Within grace period - log only - 3 persona run(s), all pass
- `SCN_02` Notify and confirm new ETA - 3 persona run(s), all pass
- `SCN_03` Merchant goes unresponsive - 3 persona run(s), all pass
- `SCN_04` Gold auto-reassignment succeeds - 3 persona run(s), all pass
- `SCN_05` Gold reassignment - no captain available - 3 persona run(s), all pass
- `SCN_06` Transient tool failure then success - 3 persona run(s), all pass
- `SCN_07` Partial failure - reassigned but notification fails - 3 persona run(s), all pass
- `SCN_08` Captain cancels while the reassignment is in flight - 3 persona run(s), all pass
- `SCN_09` ETA update crosses the escalation threshold mid-conversation - 3 persona run(s), all pass
- `SCN_10` Active outage suppresses reassignment - 3 persona run(s), all pass
- `SCN_11` ETA worsens while the agent is composing its reply - 3 persona run(s), all pass

## Policy authoring gaps

Flagged for human sign-off - the source policy does not specify these, so the
behaviour below is an adopted decision rather than a stated rule:

- 40-min boundary undefined in source policy; adopted <=40 -> R3/R4, >40 -> R5.
- R4 cancellation branch unspecified in source policy; adopted mandatory escalation.
- active_system_overrides handling absent from source policy; adopted override_map (active_outage suppresses reassignment + attach-to-incident).
- R6 says a human request overrides all other rules 'at any point', but the source policy does not say whether a session that has already RESOLVED or ESCALATED is still in scope. Adopted: terminal sessions are inert - a late human request or cancellation is recorded but starts nothing. Rationale: escalation is a one-way latch and a resolved order is closed; in production a new merchant message opens a new session rather than reviving a closed one. Flagged because a literal reading of 'at any point' would instead require reviving it.
- ORDER_PREP_COMPLETED is listed as a mid-call event but the source policy does not say what it means for the delay conversation. Adopted: it closes the session as resolved. Flagged because it is debatable - the kitchen finishing does not by itself fix a captain/delivery delay, and an alternative reading is that it only updates context and the delay rules keep applying.
- R3's auto-reassignment completes before the merchant can send a single message. Treating that as terminal would leave a Gold merchant in the 20-40 band with no in-session route to a human at all, which contradicts R6's 'at any point'. Adopted: a completed reassignment notifies and returns to monitoring rather than closing, so R6 and the rest of the policy keep applying.
