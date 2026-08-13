# Care Agent - Evaluation Report

**Result: PASS** - 30/30 runs passed (10/10 scenarios clean).

- Generated: 2026-08-13T16:22:48+00:00
- Policy version: 1
- Unauthorized promises that reached a merchant: **0**
- Unauthorized promises blocked before sending: 0
- Subjective (LLM judge) pass: skipped - no API key

Objective metrics are decided by the policy engine and structural invariants, not by a
model. The judge only grades what has no exact answer.

## Runs

| Scenario | Persona | Pass | Trajectory | Policy | Idempotent | Violations | Turns | ms |
|---|---|:--:|--:|--:|:--:|--:|--:|--:|
| SCN_01 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_01 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_01 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_02 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 3 | 1 |
| SCN_02 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 3 | 1 |
| SCN_02 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 0 |
| SCN_03 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_03 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_03 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_04 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_04 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_04 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_05 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_05 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_05 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_06 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_06 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_06 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_07 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_07 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_07 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_08 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 3 | 2 |
| SCN_08 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 1 | 1 |
| SCN_08 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 1 | 0 |
| SCN_09 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_09 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_09 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 0 | 0 |
| SCN_10 | aggressive_negotiator | PASS | 1.00 | 1.00 | yes | 0 | 3 | 1 |
| SCN_10 | indecisive_switcher | PASS | 1.00 | 1.00 | yes | 0 | 3 | 1 |
| SCN_10 | uncovered_intent | PASS | 1.00 | 1.00 | yes | 0 | 3 | 1 |

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

## Policy authoring gaps

Flagged for human sign-off - the source policy does not specify these, so the
behaviour below is an adopted decision rather than a stated rule:

- 40-min boundary undefined in source policy; adopted <=40 -> R3/R4, >40 -> R5.
- R4 cancellation branch unspecified in source policy; adopted mandatory escalation.
- active_system_overrides handling absent from source policy; adopted override_map (active_outage suppresses reassignment + attach-to-incident).
