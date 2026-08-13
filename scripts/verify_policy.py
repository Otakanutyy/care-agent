#!/usr/bin/env python
"""Validate the policy pack: JSON-Schema shape + safety invariants.

Exit code 0 when the policy is valid, 1 otherwise. Output is ASCII-only so it
renders correctly on any console encoding (e.g. Windows cp1251/cp437).

Usage:
    python scripts/verify_policy.py [POLICY_JSON] [--schema SCHEMA_JSON]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Never crash on a non-ASCII path/character under a legacy console codepage
# (e.g. Windows cp1251/cp437): replace unencodable chars instead of raising.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from care_agent.policy.loader import load_policy_dict, verify  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Careem Care Agent policy pack.")
    parser.add_argument(
        "policy",
        nargs="?",
        default=str(REPO_ROOT / "policy" / "policy.json"),
        help="path to policy.json (default: policy/policy.json)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="path to policy.schema.json (default: alongside the policy file)",
    )
    args = parser.parse_args(argv)

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"[ERROR] policy file not found: {policy_path}")
        return 1

    schema_path = Path(args.schema) if args.schema else policy_path.parent / "policy.schema.json"

    try:
        policy = load_policy_dict(policy_path)
    except Exception as exc:  # noqa: BLE001 - surface any parse error cleanly
        print(f"[ERROR] could not parse {policy_path}: {exc}")
        return 1

    results = verify(policy, schema_path=schema_path)

    print(f"Verifying policy pack: {policy_path}")
    print("-" * 68)
    width = max(len(r.name) for r in results)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name.ljust(width)}  {result.detail}")

    gaps = policy.get("authoring_gaps", [])
    if gaps:
        print("-" * 68)
        for gap in gaps:
            print(f"[NOTE] authoring gap flagged for human sign-off: {gap}")

    print("-" * 68)
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)} checks failed)")
        return 1
    print(f"RESULT: PASS ({len(results)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
