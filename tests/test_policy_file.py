"""Step 1 tests: the shipped policy pack loads, verifies, and is immutable;
and the invariant checks actually catch broken policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from care_agent.policy.loader import (
    PolicyError,
    PolicySnapshot,
    check_invariants,
    load_policy,
    load_policy_dict,
    verify,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy" / "policy.json"
SCHEMA_PATH = REPO_ROOT / "policy" / "policy.schema.json"


# --- The shipped policy is valid ---------------------------------------------


def test_real_policy_loads_and_verifies():
    snapshot = load_policy(POLICY_PATH)
    assert isinstance(snapshot, PolicySnapshot)
    results = verify(load_policy_dict(POLICY_PATH), schema_path=SCHEMA_PATH)
    failures = [r for r in results if not r.ok]
    assert not failures, failures


def test_every_invariant_passes_on_real_policy():
    results = check_invariants(load_policy_dict(POLICY_PATH))
    names = {r.name for r in results}
    # All six semantic invariants are present and passing.
    assert names == {
        "delay-coverage",
        "first-hit-ordering",
        "r6-preemption",
        "boundary-40min",
        "override-map",
        "scalars-precedence",
    }
    assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_snapshot_exposes_expected_fields():
    snapshot = load_policy(POLICY_PATH)
    assert snapshot.version == 1
    assert snapshot.retry_cap == 2
    assert snapshot.loop_guard_threshold == 3
    assert snapshot.unresponsive_attempts == 2
    assert snapshot.boolean_precedence[0] == "requests_human"
    assert snapshot.rule_by_id("R3")["action"] == "auto_reassign"
    assert snapshot.rule_by_id("R6") is None  # R6 is preemption, not a rule


# --- Immutability -------------------------------------------------------------


def test_snapshot_mapping_is_read_only():
    snapshot = load_policy(POLICY_PATH)
    with pytest.raises(TypeError):
        snapshot.raw["retry_cap"] = 999  # MappingProxyType is read-only


def test_snapshot_nested_containers_are_frozen():
    snapshot = load_policy(POLICY_PATH)
    # rules became a tuple; individual rules became read-only mappings
    assert isinstance(snapshot.rules, tuple)
    with pytest.raises(TypeError):
        snapshot.rules[0]["action"] = "escalate_immediate"


# --- Negative cases: the checks must catch breakage ---------------------------


def _invariant_failures(mutate) -> list:
    policy = load_policy_dict(POLICY_PATH)
    mutate(policy)
    return [r for r in check_invariants(policy) if not r.ok]


def test_gap_in_coverage_detected():
    def mutate(p):
        p["rules"] = [r for r in p["rules"] if r["id"] != "R2"]  # leaves a 10..20 gap

    failures = _invariant_failures(mutate)
    assert any(r.name == "delay-coverage" for r in failures)


def test_r6_in_rules_detected():
    def mutate(p):
        p["rules"].append(
            {"id": "R6", "delay": [0, None], "tier": "*", "action": "escalate_immediate", "escalates": "always"}
        )

    failures = _invariant_failures(mutate)
    assert any(r.name == "r6-preemption" for r in failures)


def test_missing_boundary_detected():
    def mutate(p):
        del p["boundary_inclusivity"]["R3_R4_upper"]

    failures = _invariant_failures(mutate)
    assert any(r.name == "boundary-40min" for r in failures)


def test_boundary_overlap_detected():
    def mutate(p):
        p["boundary_inclusivity"]["R3_R4_upper"] = "inclusive"
        p["boundary_inclusivity"]["R5_lower"] = "inclusive"  # both claim 40 -> overlap

    failures = _invariant_failures(mutate)
    assert any(r.name == "boundary-40min" for r in failures)


def test_override_suppresses_unknown_rule_detected():
    def mutate(p):
        p["override_map"]["active_outage"]["suppressed_rules"] = ["R9"]

    failures = _invariant_failures(mutate)
    assert any(r.name == "override-map" for r in failures)


def test_precedence_not_led_by_requests_human_detected():
    def mutate(p):
        p["boolean_precedence"] = [
            "requests_cancellation",
            "requests_human",
            "accepts_reassignment",
            "prefers_to_wait",
            "confirms_new_eta",
        ]

    failures = _invariant_failures(mutate)
    assert any(r.name == "scalars-precedence" for r in failures)


def test_load_policy_raises_on_broken_file(tmp_path):
    policy = load_policy_dict(POLICY_PATH)
    policy["rules"] = [r for r in policy["rules"] if r["id"] != "R2"]  # coverage gap
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(broken, schema_path=SCHEMA_PATH)
