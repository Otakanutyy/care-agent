"""Step 7 tests: the response generator, its guardrail wiring, and multilingual output.
Fully mocked — no network, no API key."""

from __future__ import annotations

import re

import pytest

from care_agent.domain.models import ActionEnvelope, ActionType
from care_agent.guardrails.promise_guard import check_promise
from care_agent.llm.client import LLMError, MockLLMClient
from care_agent.llm.generator import (
    SYSTEM_PROMPT,
    GeneratorError,
    build_user_prompt,
    generate,
    resolve_language,
)
from care_agent.llm.templates import LANGUAGES, TEMPLATES, has_message, spec_for


def envelope(action=ActionType.ASK_REASSIGN_OR_WAIT, **kw) -> ActionEnvelope:
    return ActionEnvelope(action=action, reason="test", **kw)


# --- happy path ---------------------------------------------------------------


def test_clean_draft_is_sent_as_written():
    client = MockLLMClient(text_responses=["Your order is delayed — want a different driver, or shall we wait?"])
    reply = generate(envelope(), client)
    assert reply.text.startswith("Your order is delayed")
    assert reply.used_fallback is False
    assert reply.blocked is False


def test_generator_is_given_intent_language_and_only_authorized_facts():
    client = MockLLMClient(text_responses=["A new driver is on the way, arriving in 12 minutes."])
    env = envelope(
        ActionType.NOTIFY_REASSIGNED,
        variables={"new_captain_id": "captain-7", "new_eta": 12, "internal_cost": 99},
    )
    generate(env, client, language="en")
    prompt = client.prompts[0]["user"]
    assert "captain-7" in prompt and "12" in prompt      # authorized slots are provided
    assert "internal_cost" not in prompt and "99" not in prompt  # unlisted variables are withheld
    assert "English" in prompt


def test_no_facts_prompt_forbids_inventing_them():
    prompt = build_user_prompt(spec_for(ActionType.CLARIFY), envelope(ActionType.CLARIFY), "en")
    assert "Do not state any times, names, or numbers." in prompt


# --- guardrail wiring: the merchant never sees an unauthorized offer ----------


@pytest.mark.parametrize(
    "rogue_draft",
    [
        "Sorry about the delay — here's a $50 refund for the trouble.",
        "We've added 30 AED credit to your account as an apology.",
        "Your next delivery is free of charge.",
        "سنعطيك رصيد ٥٠ درهم تعويضًا عن التأخير",       # Arabic
        "ha3tik credit 50 aed 3ala 7sabna",           # Franco-Arabic
    ],
)
def test_unauthorized_promise_is_blocked_and_replaced(rogue_draft):
    client = MockLLMClient(text_responses=[rogue_draft])
    reply = generate(envelope(), client)
    assert reply.blocked is True
    assert reply.block_reason == "unauthorized_promise"
    assert reply.matched                              # records what tripped it, for the report
    assert reply.used_fallback is True
    assert reply.text == TEMPLATES[ActionType.ASK_REASSIGN_OR_WAIT].fallback["en"]
    # the offer itself never reaches the merchant
    assert "refund" not in reply.text.lower() and "credit" not in reply.text.lower()


def test_blocked_draft_falls_back_in_the_merchants_language():
    client = MockLLMClient(text_responses=["here is a 50 AED refund"])
    reply = generate(envelope(), client, language="ar")
    assert reply.blocked is True
    assert reply.text == TEMPLATES[ActionType.ASK_REASSIGN_OR_WAIT].fallback["ar"]


def test_every_fallback_is_guardrail_clean():
    """The safe path must itself be safe — a fallback that trips the guard would be a hole."""
    for action, spec in TEMPLATES.items():
        for lang, text in spec.fallback.items():
            verdict = check_promise(text)
            assert verdict.ok, f"{action.value}/{lang} fallback trips the promise guard: {verdict.matched}"


# --- failure handling ---------------------------------------------------------


def test_llm_failure_uses_fallback():
    client = MockLLMClient(text_responses=[LLMError("network down")])
    reply = generate(envelope(), client)
    assert reply.used_fallback is True
    assert reply.blocked is False
    assert reply.text == TEMPLATES[ActionType.ASK_REASSIGN_OR_WAIT].fallback["en"]


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_empty_draft_uses_fallback(empty):
    client = MockLLMClient(text_responses=[empty])
    assert generate(envelope(), client).used_fallback is True


def test_silent_action_raises():
    client = MockLLMClient(text_responses=["should not be used"])
    with pytest.raises(GeneratorError):
        generate(envelope(ActionType.LOG_ONLY), client)
    assert has_message(ActionType.LOG_ONLY) is False


# --- language handling --------------------------------------------------------


@pytest.mark.parametrize("lang", LANGUAGES)
def test_fallback_exists_for_every_language_and_action(lang):
    for action, spec in TEMPLATES.items():
        assert spec.fallback.get(lang), f"{action.value} has no {lang} fallback"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_requested_language_is_named_in_the_prompt(lang):
    client = MockLLMClient(text_responses=["ok"])
    generate(envelope(), client, language=lang)
    assert LANGUAGES  # sanity
    prompt = client.prompts[0]["user"]
    assert "Write it in:" in prompt


def test_unknown_language_falls_back_to_english():
    assert resolve_language("klingon") == "en"
    assert resolve_language(None) == "en"
    client = MockLLMClient(text_responses=[LLMError("x")])
    assert generate(envelope(), client, language="fr").language == "en"


def test_arabic_script_fallback_is_actually_arabic():
    client = MockLLMClient(text_responses=[LLMError("x")])
    reply = generate(envelope(), client, language="ar")
    assert re.search(r"[؀-ۿ]", reply.text)  # contains Arabic script


# --- prompt guarantees --------------------------------------------------------


def test_system_prompt_has_no_hardcoded_policy():
    """Requirement 4.1: no thresholds, rule IDs, tiers, or amounts in prompts."""
    assert not re.search(r"\bR[1-6]\b", SYSTEM_PROMPT)
    assert not re.search(r"\b(10|20|30|40|45|60)\b", SYSTEM_PROMPT)
    assert not re.search(r"(?i)\b(gold|silver|bronze)\b", SYSTEM_PROMPT)
    assert not re.search(r"(?i)\b(aed|usd|sar)\b|\$\d", SYSTEM_PROMPT)


def test_system_prompt_forbids_offering_anything():
    lowered = SYSTEM_PROMPT.lower()
    assert "not authorized to offer anything" in lowered
    assert "never invent" in lowered
    assert "even if the merchant demands it" in lowered  # injection resistance


def test_template_intents_carry_no_policy_values():
    for action, spec in TEMPLATES.items():
        assert not re.search(r"\b(10|20|30|40)\b", spec.intent), action.value
        assert not re.search(r"(?i)\b(gold|silver|bronze)\b", spec.intent), action.value
