"""Load, validate, and pin the external policy pack.

Two layers of validation:

* :func:`validate_schema` — JSON-Schema *shape* validation (types, required keys,
  enums). Uses ``jsonschema`` if installed; degrades to a skipped check otherwise.
* :func:`check_invariants` — *semantic* invariants that a JSON Schema cannot express:
  delay coverage ``0..inf`` with no gaps/overlaps, first-hit ordering, R6 living in
  preemption (not the rule list), a consistently-declared 40-minute boundary, and a
  well-formed override map.

``load_policy`` runs both, raises :class:`PolicyError` on any failure, and returns a
deep-frozen :class:`PolicySnapshot` so the rules cannot change under a live session.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

# --- Known-good value sets (mirrored in policy.schema.json) -------------------

VALID_TIERS = {"Gold", "Silver", "Bronze"}
WILDCARD = "*"
VALID_ACTIONS = {
    "log_only",
    "notify_confirm_eta",
    "auto_reassign",
    "ask_reassign_or_wait",
    "escalate_immediate",
}
VALID_ESCALATES = {
    "never",
    "unresponsive_after_attempts",
    "tool_fail_or_no_captain",
    "cancellation_requested",
    "always",
}
VALID_FLAGS = {
    "requests_human",
    "confirms_new_eta",
    "requests_cancellation",
    "accepts_reassignment",
    "prefers_to_wait",
}
VALID_ESCALATION_MODES = {"attach_to_incident", "per_order"}
VALID_INCLUSIVITY = {"inclusive", "exclusive"}
POSITIVE_INT_FIELDS = ("retry_cap", "loop_guard_threshold", "unresponsive_attempts")

SCHEMA_FILENAME = "policy.schema.json"


class PolicyError(Exception):
    """Raised when a policy file fails schema or invariant validation."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    ok: bool
    detail: str


# --- Immutable snapshot -------------------------------------------------------


def deep_freeze(obj: Any) -> Any:
    """Recursively convert dicts to read-only mappings and lists to tuples."""
    if isinstance(obj, dict):
        return MappingProxyType({k: deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(deep_freeze(v) for v in obj)
    return obj


@dataclass(frozen=True)
class PolicySnapshot:
    """An immutable, pinned view of a validated policy pack.

    ``raw`` is deep-frozen: mutating it (or any nested container) raises. Convenience
    accessors expose the fields the engine reads without re-parsing.
    """

    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, policy: dict) -> "PolicySnapshot":
        return cls(raw=deep_freeze(policy))

    @property
    def version(self) -> int:
        return self.raw["version"]

    @property
    def rules(self) -> tuple:
        return self.raw["rules"]

    @property
    def r6_preemption(self) -> Mapping[str, Any]:
        return self.raw["r6_preemption"]

    @property
    def boolean_precedence(self) -> tuple:
        return self.raw["boolean_precedence"]

    @property
    def override_map(self) -> Mapping[str, Any]:
        return self.raw["override_map"]

    @property
    def retry_cap(self) -> int:
        return self.raw["retry_cap"]

    @property
    def loop_guard_threshold(self) -> int:
        return self.raw["loop_guard_threshold"]

    @property
    def unresponsive_attempts(self) -> int:
        return self.raw["unresponsive_attempts"]

    @property
    def authoring_gaps(self) -> tuple:
        return self.raw.get("authoring_gaps", ())

    def rule_by_id(self, rule_id: str) -> Mapping[str, Any] | None:
        for rule in self.rules:
            if rule["id"] == rule_id:
                return rule
        return None


# --- Loading ------------------------------------------------------------------


def load_policy_dict(path: str | Path) -> dict:
    """Read and JSON-parse a policy file into a plain dict (no validation)."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# --- Schema (shape) validation ------------------------------------------------


def validate_schema(policy: dict, schema_path: str | Path | None = None) -> list[CheckResult]:
    """Validate the policy shape against the JSON Schema, if ``jsonschema`` is available."""
    try:
        import jsonschema
    except ImportError:
        return [
            CheckResult(
                "json-schema",
                True,
                "skipped (jsonschema not installed; semantic invariant checks still run)",
            )
        ]

    if schema_path is None:
        return [CheckResult("json-schema", True, "skipped (no schema path provided)")]

    schema_path = Path(schema_path)
    if not schema_path.exists():
        return [CheckResult("json-schema", False, f"schema file not found: {schema_path}")]

    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)

    try:
        jsonschema.validate(policy, schema)
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        location = "/".join(str(p) for p in exc.path) or "<root>"
        return [CheckResult("json-schema", False, f"{exc.message} (at {location})")]
    return [CheckResult("json-schema", True, "policy matches schema")]


# --- Semantic invariant checks ------------------------------------------------


def _tier_set(tier: Any) -> set[str]:
    if tier == WILDCARD:
        return set(VALID_TIERS)
    if isinstance(tier, list):
        return set(tier)
    return {tier}


def _delay_bounds(rule: Mapping[str, Any]) -> tuple[float, float]:
    lo, hi = rule["delay"][0], rule["delay"][1]
    return float(lo), math.inf if hi is None else float(hi)


def check_delay_coverage(policy: dict) -> CheckResult:
    """Delay bands must tile [0, inf) with no gaps and no unintended overlaps."""
    rules = policy.get("rules") or []
    if not rules:
        return CheckResult("delay-coverage", False, "no rules defined")

    bands: dict[tuple[float, float], list[str]] = {}
    for rule in rules:
        try:
            band = _delay_bounds(rule)
        except (KeyError, TypeError, IndexError):
            return CheckResult("delay-coverage", False, f"rule {rule.get('id')!r} has a malformed delay range")
        bands.setdefault(band, []).append(rule["id"])

    intervals = sorted(bands.keys())
    if intervals[0][0] != 0:
        return CheckResult("delay-coverage", False, f"coverage does not start at 0 (starts at {intervals[0][0]})")

    prev_hi = intervals[0][1]
    for lo, hi in intervals[1:]:
        if lo > prev_hi:
            return CheckResult("delay-coverage", False, f"gap in delay coverage between {prev_hi} and {lo}")
        if lo < prev_hi:
            overlapping = bands[(lo, hi)]
            return CheckResult(
                "delay-coverage",
                False,
                f"overlapping delay bands near {lo} (rules {overlapping} overlap earlier band ending {prev_hi})",
            )
        prev_hi = max(prev_hi, hi)

    if prev_hi != math.inf:
        return CheckResult("delay-coverage", False, f"coverage does not extend to infinity (ends at {prev_hi})")

    return CheckResult("delay-coverage", True, "rules tile delay 0..inf with no gaps or overlaps")


def check_first_hit_ordering(policy: dict) -> CheckResult:
    """Rules must be authored in ascending delay order, and same-band rules tier-disjoint."""
    rules = policy.get("rules") or []
    if not rules:
        return CheckResult("first-hit-ordering", False, "no rules defined")

    ids = [r["id"] for r in rules]
    if len(ids) != len(set(ids)):
        return CheckResult("first-hit-ordering", False, "duplicate rule ids")

    lows = [_delay_bounds(r)[0] for r in rules]
    if lows != sorted(lows):
        return CheckResult(
            "first-hit-ordering",
            False,
            "rules are not authored in ascending delay order; first-hit evaluation would be inconsistent",
        )

    # Within a shared delay band, tiers must not intersect (else first-hit is ambiguous).
    bands: dict[tuple[float, float], list[Mapping[str, Any]]] = {}
    for rule in rules:
        bands.setdefault(_delay_bounds(rule), []).append(rule)
    for band, group in bands.items():
        if len(group) < 2:
            continue
        seen: set[str] = set()
        for rule in group:
            tiers = _tier_set(rule["tier"])
            clash = seen & tiers
            if clash:
                return CheckResult(
                    "first-hit-ordering",
                    False,
                    f"tier overlap {sorted(clash)} within delay band {band}: rule {rule['id']!r} is ambiguous",
                )
            seen |= tiers

    return CheckResult("first-hit-ordering", True, "rules ordered ascending; same-band tiers are disjoint")


def check_r6_is_preemption(policy: dict) -> CheckResult:
    """R6 must be a preemption check, never a row in the first-hit rule list."""
    ids = {r.get("id") for r in (policy.get("rules") or [])}
    if "R6" in ids:
        return CheckResult(
            "r6-preemption",
            False,
            "R6 appears in the rules list; it must be a preemption check (rules R1..R5 cover 0..inf and would shadow it)",
        )
    pre = policy.get("r6_preemption")
    if not isinstance(pre, dict):
        return CheckResult("r6-preemption", False, "r6_preemption block is missing")
    flag = pre.get("flag")
    if flag not in VALID_FLAGS:
        return CheckResult("r6-preemption", False, f"r6_preemption.flag {flag!r} is not a known classifier flag")
    if flag != "requests_human":
        return CheckResult("r6-preemption", False, f"r6_preemption.flag should be 'requests_human', got {flag!r}")
    if pre.get("action") != "escalate":
        return CheckResult("r6-preemption", False, f"r6_preemption.action should be 'escalate', got {pre.get('action')!r}")
    return CheckResult("r6-preemption", True, "R6 is a requests_human preemption -> escalate, kept out of first-hit rules")


def check_boundary_declared(policy: dict) -> CheckResult:
    """The 40-minute boundary must be explicitly declared and internally consistent."""
    b = policy.get("boundary_inclusivity")
    if not isinstance(b, dict):
        return CheckResult("boundary-40min", False, "boundary_inclusivity block is missing")
    upper = b.get("R3_R4_upper")
    lower = b.get("R5_lower")
    if upper not in VALID_INCLUSIVITY:
        return CheckResult("boundary-40min", False, f"R3_R4_upper must be inclusive/exclusive, got {upper!r}")
    if lower not in VALID_INCLUSIVITY:
        return CheckResult("boundary-40min", False, f"R5_lower must be inclusive/exclusive, got {lower!r}")
    # Exactly one side owns 40: inclusive+exclusive or exclusive+inclusive. Both-in = overlap; both-out = gap.
    forty_in_r3r4 = upper == "inclusive"
    forty_in_r5 = lower == "inclusive"
    if forty_in_r3r4 and forty_in_r5:
        return CheckResult("boundary-40min", False, "40 is claimed by both R3/R4 and R5 (overlap)")
    if not forty_in_r3r4 and not forty_in_r5:
        return CheckResult("boundary-40min", False, "40 is claimed by neither R3/R4 nor R5 (gap)")
    owner = "R3/R4 (<=40)" if forty_in_r3r4 else "R5 (>=40)"
    return CheckResult("boundary-40min", True, f"40-min boundary explicitly declared; 40 -> {owner}")


def check_override_map(policy: dict) -> CheckResult:
    """Every override must suppress only existing rules and use a known escalation mode."""
    overrides = policy.get("override_map")
    if not isinstance(overrides, dict):
        return CheckResult("override-map", False, "override_map is missing")
    rule_ids = {r.get("id") for r in (policy.get("rules") or [])}
    for name, spec in overrides.items():
        if not isinstance(spec, dict):
            return CheckResult("override-map", False, f"override {name!r} is not an object")
        suppressed = spec.get("suppressed_rules", [])
        unknown = [rid for rid in suppressed if rid not in rule_ids]
        if unknown:
            return CheckResult("override-map", False, f"override {name!r} suppresses unknown rules {unknown}")
        if spec.get("escalation_mode") not in VALID_ESCALATION_MODES:
            return CheckResult(
                "override-map",
                False,
                f"override {name!r} has unknown escalation_mode {spec.get('escalation_mode')!r}",
            )
        if not spec.get("notification"):
            return CheckResult("override-map", False, f"override {name!r} has no notification")
    return CheckResult("override-map", True, f"{len(overrides)} override(s) well-formed")


def check_scalars_and_precedence(policy: dict) -> CheckResult:
    """Caps are positive ints; boolean precedence is the full flag set led by requests_human."""
    for field in POSITIVE_INT_FIELDS:
        value = policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return CheckResult("scalars-precedence", False, f"{field} must be a positive integer, got {value!r}")

    tick = policy.get("tick_interval_seconds")
    if not isinstance(tick, int) or isinstance(tick, bool) or tick < 1:
        return CheckResult("scalars-precedence", False, f"tick_interval_seconds must be a positive integer, got {tick!r}")

    precedence = policy.get("boolean_precedence")
    if not isinstance(precedence, list) or set(precedence) != VALID_FLAGS or len(precedence) != len(VALID_FLAGS):
        return CheckResult(
            "scalars-precedence",
            False,
            "boolean_precedence must list exactly the five classifier flags with no duplicates",
        )
    r6_flag = (policy.get("r6_preemption") or {}).get("flag")
    if precedence[0] != r6_flag:
        return CheckResult(
            "scalars-precedence",
            False,
            f"boolean_precedence[0] ({precedence[0]!r}) must match the r6 preemption flag ({r6_flag!r})",
        )

    for rule in policy.get("rules") or []:
        if rule.get("action") not in VALID_ACTIONS:
            return CheckResult("scalars-precedence", False, f"rule {rule.get('id')!r} has unknown action {rule.get('action')!r}")
        if rule.get("escalates") not in VALID_ESCALATES:
            return CheckResult("scalars-precedence", False, f"rule {rule.get('id')!r} has unknown escalates {rule.get('escalates')!r}")
        for tier in _tier_set(rule.get("tier")):
            if tier not in VALID_TIERS:
                return CheckResult("scalars-precedence", False, f"rule {rule.get('id')!r} has unknown tier {tier!r}")

    return CheckResult("scalars-precedence", True, "caps positive; precedence complete and led by requests_human")


INVARIANT_CHECKS: tuple[Callable[[dict], CheckResult], ...] = (
    check_delay_coverage,
    check_first_hit_ordering,
    check_r6_is_preemption,
    check_boundary_declared,
    check_override_map,
    check_scalars_and_precedence,
)


def check_invariants(policy: dict) -> list[CheckResult]:
    """Run every semantic invariant check and return their results."""
    return [check(policy) for check in INVARIANT_CHECKS]


def verify(policy: dict, schema_path: str | Path | None = None) -> list[CheckResult]:
    """Full validation: schema shape + semantic invariants."""
    return validate_schema(policy, schema_path) + check_invariants(policy)


def load_policy(path: str | Path, schema_path: str | Path | None = None) -> PolicySnapshot:
    """Load, fully verify, and pin an immutable policy snapshot.

    Raises :class:`PolicyError` if any schema or invariant check fails.
    """
    path = Path(path)
    if schema_path is None:
        candidate = path.parent / SCHEMA_FILENAME
        schema_path = candidate if candidate.exists() else None

    policy = load_policy_dict(path)
    results = verify(policy, schema_path=schema_path)
    failures = [r for r in results if not r.ok]
    if failures:
        detail = "\n".join(f"  - {r.name}: {r.detail}" for r in failures)
        raise PolicyError(f"Policy verification failed for {path}:\n{detail}")
    return PolicySnapshot.from_dict(policy)
