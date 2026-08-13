"""Markdown rendering for MCP tool results.

When a reviewer drives this agent through their own Claude, **the chat window is the UI**. MCP
tool results carry text content that the client renders, so a decision comes back as a formatted
card rather than a JSON blob the reviewer has to parse by eye.

Each card answers the question the whole submission rests on — *what decided this, the policy or
the model?* — by putting the rule id, the reason, and the financial-tool count next to the
message they produced. The structured payload travels alongside it untouched, so an agent
driving this programmatically still gets clean data.
"""

from __future__ import annotations

from typing import Any

# Financial tools deliberately absent from the engine's action vocabulary. Named here so every
# card can state positively that they were not called, rather than leaving it to be inferred.
FINANCIAL_TOOLS = ("cancel_order", "issue_merchant_credit")

_RULE_NOTES = {
    "R6": "preemption — evaluated *before* the rule list, so it overrides everything",
}


def _esc(text: str) -> str:
    """Keep merchant-supplied text from breaking out of a table cell or blockquote."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def _guardrail_badge(status: str | None) -> str:
    return {"clean": "PASS - clean", "blocked": "BLOCKED - draft discarded"}.get(
        status or "", status or "n/a"
    )


def _financial_line(tool_calls: dict[str, Any]) -> str:
    hits = {name: n for name, n in (tool_calls or {}).items() if name in FINANCIAL_TOOLS}
    if hits:
        return f"**Financial tools:** {hits} <- UNEXPECTED, these should be unreachable"
    return (
        "**Financial tools:** none called - `cancel_order` and `issue_merchant_credit` are not "
        "in the engine's action vocabulary at all"
    )


def _tools_line(tool_calls: dict[str, Any]) -> str:
    if not tool_calls:
        return "**Tool calls:** none yet"
    rendered = ", ".join(f"`{name}` x{count}" for name, count in sorted(tool_calls.items()))
    return f"**Tool calls (cumulative):** {rendered}"


def turn_card(payload: dict[str, Any], merchant_said: str | None = None) -> str:
    """One turn, rendered as a decision card."""
    if payload.get("error"):
        return f"**{payload['error']}**\n\n{payload.get('detail', '')}"

    rule = payload.get("matched_rule")
    action = payload.get("action")
    lines: list[str] = []

    header = f"### {action or 'no action'}" + (f"  <-  rule **{rule}**" if rule else "")
    lines.append(header)

    if merchant_said:
        lines.append(f"\n> **Merchant:** {_esc(merchant_said)}")

    for reply in payload.get("agent_replies") or []:
        flag = " *(guardrail blocked this draft)*" if reply.get("blocked") else ""
        lines.append(f"\n**Agent:** {_esc(reply.get('text', ''))}{flag}")

    note = _RULE_NOTES.get(rule or "")
    rule_cell = f"`{rule}`" + (f" - {note}" if note else "") if rule else "n/a"
    transition = payload.get("fsm_state") or "n/a"
    if payload.get("previous_fsm_state") and payload["previous_fsm_state"] != transition:
        transition = f"`{payload['previous_fsm_state']}` -> `{transition}`"
    else:
        transition = f"`{transition}`"

    lines.append("\n| | |\n|---|---|")
    lines.append(f"| **Rule that decided** | {rule_cell} |")
    lines.append(f"| **Action** | `{action}` |")
    lines.append(f"| **Reason** | `{payload.get('decision_reason')}` |")
    lines.append(f"| **Guardrail** | {_guardrail_badge(payload.get('guardrail_status'))} |")
    lines.append(f"| **FSM state** | {transition} |")
    if payload.get("delay_minutes") is not None:
        lines.append(f"| **Delay** | {payload['delay_minutes']} min |")
    if payload.get("active_system_overrides"):
        lines.append(f"| **Overrides active** | `{payload['active_system_overrides']}` |")

    ticket = payload.get("escalation")
    if ticket:
        halt = " (guardrail halt)" if ticket.get("guardrail_halt") else ""
        lines.append(
            f"\n**Escalated to a human** - ticket `{ticket.get('ticket_id')}`, "
            f"reason `{ticket.get('reason')}`, mode `{ticket.get('escalation_mode')}`{halt}. "
            "The ticket carries a full context snapshot so the human does not inherit an angry "
            "merchant and no history."
        )

    tool_calls = payload.get("tool_calls") or {}
    lines.append(f"\n{_tools_line(tool_calls)}")
    lines.append(_financial_line(tool_calls))

    if payload.get("note"):
        lines.append(f"\n*{payload['note']}*")

    return "\n".join(lines)


def session_card(payload: dict[str, Any]) -> str:
    """The opening of a session: context plus the agent's proactive first message."""
    if payload.get("error"):
        return f"**{payload['error']}**\n\n{payload.get('detail', '')}"

    context = payload.get("context") or {}
    header = [
        f"## Session `{payload.get('session_id')}`",
        "",
        f"**{context.get('merchant_tier')}** merchant, **{context.get('delay_minutes')} min** "
        f"delay, order `{context.get('order_id')}`"
        + (f", overrides `{context['active_system_overrides']}`" if context.get("active_system_overrides") else ""),
        "",
        f"*mode: {payload.get('mode')} · policy v{payload.get('policy_version')}*",
        "",
        "The agent opens the conversation itself - it is proactive, not a chatbot waiting to be "
        "asked. The rule below chose that opening.",
        "",
    ]
    return "\n".join(header) + turn_card(payload)


def _trajectory_line(payload: dict[str, Any]) -> str:
    """The sequence of *policy actions*, which is what the trajectory metric scores.

    Deliberately not the raw effect log — internal effects like `log` and `resolve` are
    machinery, and mixing them in makes the decision sequence harder to read.
    """
    rules: dict[str, str] = {}
    for step in payload.get("trace") or []:
        if step.get("action") and step.get("rule_id"):
            rules.setdefault(step["action"], step["rule_id"])

    steps = [
        f"{action} ({rules[action]})" if action in rules else action
        for action in payload.get("trajectory") or []
    ]
    end = payload.get("fsm_state") or "?"
    return " -> ".join(["INIT", *steps, f"[{end}]"])


def trace_card(payload: dict[str, Any]) -> str:
    """The full decision record — the evidence for everything the agent did."""
    if payload.get("error"):
        return f"**{payload['error']}**\n\n{payload.get('detail', '')}"

    lines = [
        f"## Decision record - session `{payload.get('session_id')}`",
        "",
        "### Conversation",
        "",
    ]

    for entry in payload.get("transcript") or []:
        who = "Merchant" if entry.get("speaker") == "merchant" else "Agent"
        text = _esc(entry.get("text", ""))
        if who == "Agent":
            tag = f" `[{entry.get('action')} / {entry.get('rule_id')}]`" if entry.get("action") else ""
            blocked = " **(BLOCKED)**" if entry.get("blocked") else ""
            lines.append(f"- **{who}:**{tag}{blocked} {text}")
        else:
            lines.append(f"- **{who}:** {text}")

    lines += [
        "",
        "### Trajectory",
        "",
        "```",
        _trajectory_line(payload),
        "```",
        "",
        f"Final state: `{payload.get('fsm_state')}`"
        + (f" (terminal: `{payload.get('terminal_reason')}`)" if payload.get("terminal_reason") else ""),
        "",
        "### Evidence",
        "",
    ]

    call_order = payload.get("tool_call_order") or []
    lines.append(
        f"- **Tool calls in order:** {' -> '.join(f'`{c}`' for c in call_order)}"
        if call_order
        else "- **Tool calls:** none"
    )
    lines.append(f"- {_financial_line(payload.get('tool_calls') or {})}")
    blocks = payload.get("guardrail_blocks") or 0
    lines.append(
        f"- **Guardrail blocks:** {blocks}"
        + (" (drafts stopped before reaching the merchant - these are successes)" if blocks else "")
    )

    for ticket in payload.get("tickets") or []:
        lines.append(
            f"- **Escalation `{ticket.get('ticket_id')}`:** reason `{ticket.get('reason')}`, "
            f"mode `{ticket.get('escalation_mode')}`"
        )

    return "\n".join(lines)


def sessions_card(payload: dict[str, Any]) -> str:
    rows = payload.get("sessions") or []
    if not rows:
        return "No active sessions. Call `start_session` to begin."
    lines = [
        "## Active sessions",
        "",
        "| session | order | turns | state | terminal |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['session_id']}` | `{row['order_id']}` | {row['turns']} | "
            f"`{row['fsm_state']}` | {row.get('terminal_reason') or '-'} |"
        )
    limits = payload.get("limits", {})
    lines += [
        "",
        f"*mode: {payload.get('mode')} · policy v{payload.get('policy_version')} · "
        f"caps: {limits.get('max_sessions')} sessions, {limits.get('max_turns')} turns each*",
    ]
    return "\n".join(lines)
