"""Per-action message specs.

Each merchant-visible action has one spec describing:

* ``intent`` — what the reply must convey. This is what the generator is asked to phrase; it
  never contains a threshold, tier, rule ID, or amount.
* ``slots`` — the only envelope variables the generator may reference. Anything not listed is
  withheld, so the model cannot restate a value the engine did not authorize.
* ``fallback`` — a deterministic, pre-approved message per supported language, used when the
  generator fails or its draft is blocked by the guardrail. Every fallback is guardrail-clean
  by construction (a test enforces this), so the safe path can never itself leak a promise.

Actions with no entry here produce no merchant-visible message (e.g. log-only, or dispatching
a tool chain — the merchant hears about that when the result arrives).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from care_agent.domain.models import ActionType

LANGUAGES = ("en", "ar", "ar-latn")

LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic (Arabic script)",
    "ar-latn": "Franco-Arabic (Arabic written in Latin letters and numerals)",
}


@dataclass(frozen=True)
class MessageSpec:
    intent: str
    fallback: dict[str, str]
    slots: tuple[str, ...] = field(default_factory=tuple)


TEMPLATES: dict[ActionType, MessageSpec] = {
    ActionType.NOTIFY_CONFIRM_ETA: MessageSpec(
        intent=(
            "Let the merchant know their order is running late, and ask them to confirm "
            "whether the updated arrival time works for them."
        ),
        fallback={
            "en": "Your order is running late. Could you confirm whether the updated arrival time works for you?",
            "ar": "طلبك متأخر. هل يمكنك تأكيد ما إذا كان وقت الوصول الجديد مناسبًا لك؟",
            "ar-latn": "Talabak mit2akhar. Momken t2akkid iza wa2t el wusul el jdeed mnee7 ma3ak?",
        },
    ),
    ActionType.ASK_REASSIGN_OR_WAIT: MessageSpec(
        intent=(
            "Let the merchant know their order is delayed, and ask whether they would like a "
            "different driver assigned or would prefer to wait for the current one."
        ),
        fallback={
            "en": "Your order is delayed. Would you like us to assign a different driver, or would you prefer to wait for the current one?",
            "ar": "طلبك متأخر. هل تريد أن نعيّن كابتن آخر، أم تفضّل الانتظار للكابتن الحالي؟",
            "ar-latn": "Talabak mit2akhar. Bidak n3ayyen captain tani, wala btfaddel tantazer el captain el 7ali?",
        },
    ),
    ActionType.NOTIFY_REASSIGNED: MessageSpec(
        intent=(
            "Let the merchant know a different driver has now been assigned to their order and "
            "is on the way, including the new arrival time if one is given."
        ),
        slots=("new_captain_id", "new_eta"),
        fallback={
            "en": "A different driver has been assigned to your order and is on the way.",
            "ar": "تم تعيين كابتن آخر لطلبك، وهو في الطريق إليك الآن.",
            "ar-latn": "Tam ta3yeen captain tani la talabak, w howwe bel tare2 la 3andak.",
        },
    ),
    ActionType.ACKNOWLEDGE_WAIT: MessageSpec(
        intent=(
            "Acknowledge that the merchant prefers to wait, confirm the current driver stays "
            "assigned, and say you will let them know when there is an update."
        ),
        fallback={
            "en": "Understood — we'll keep the current driver assigned and let you know as soon as there's an update.",
            "ar": "تمام، سنبقي الكابتن الحالي ونعلمك فور توفر أي تحديث.",
            "ar-latn": "Tamam, ra7 nkhalli el captain el 7ali w n3almak awwal ma yseer fi update.",
        },
    ),
    ActionType.RESOLVE_ETA_CONFIRMED: MessageSpec(
        intent="Thank the merchant for confirming the updated arrival time, and close the conversation politely.",
        fallback={
            "en": "Thanks for confirming — we'll keep an eye on your order.",
            "ar": "شكرًا لتأكيدك، سنتابع طلبك.",
            "ar-latn": "Shukran 3al ta2kid, ra7 ntabe3 talabak.",
        },
    ),
    ActionType.DEGRADED_MODE_NOTICE: MessageSpec(
        intent=(
            "Let the merchant know a wider service disruption is currently affecting deliveries "
            "and that the team is working on it. Do not state or estimate an arrival time."
        ),
        fallback={
            "en": "We're currently experiencing a wider service disruption affecting deliveries. Our team is working on it and we'll update you as soon as we can.",
            "ar": "نواجه حاليًا اضطرابًا عامًا في الخدمة يؤثر على عمليات التوصيل. فريقنا يعمل على معالجته وسنوافيك بالمستجدات في أقرب وقت.",
            "ar-latn": "Halla2 fi moshkile 3amme bel khidme 3am tathir 3ala el tawsil. El team 3am yishtighil 3aleha w ra7 nkhabbrak awwal ma nfik.",
        },
    ),
    # Reached when the merchant's intent is outside what this conversation can act on — the
    # "uncovered intent" case. It must NOT claim incomprehension: the classifier usually
    # understood perfectly, there is simply no authorized action for the request. Saying "I
    # didn't catch that" is both false and reads as a broken bot. Say what is true instead, and
    # put the actual choice back. If the merchant keeps pushing, the loop guard hands off.
    ActionType.CLARIFY: MessageSpec(
        intent=(
            "Say that this is not something you can help with in this conversation, that you "
            "are following up about the delayed order, and ask how they would like to proceed. "
            "Do not name or offer any specific option."
        ),
        fallback={
            "en": "That's not something I can help with here — I'm following up about your "
                  "delayed order. How would you like to proceed?",
            "ar": "هذا ليس أمرًا يمكنني مساعدتك به هنا — أنا أتابع معك بخصوص تأخر طلبك. كيف تحب أن نكمل؟",
            "ar-latn": "Hayda mish shi fiyye sa3dak fi hon — ana 3am tabe3 ma3ak bi khusus "
                       "ta2akhur talabak. Kif btheb nkammel?",
        },
    ),
    ActionType.ESCALATE: MessageSpec(
        intent=(
            "Let the merchant know you are handing the conversation to a colleague from the "
            "team, who will follow up with them shortly."
        ),
        fallback={
            "en": "I'm connecting you with a colleague from our team who will follow up with you shortly.",
            "ar": "سأحوّلك إلى زميل من فريقنا وسيتواصل معك قريبًا.",
            "ar-latn": "Ra7 7awwlak la zameel men el team, w ra7 yitwasal ma3ak areeban.",
        },
    ),
}


def spec_for(action: ActionType) -> MessageSpec | None:
    """The message spec for an action, or None when the action is silent to the merchant."""
    return TEMPLATES.get(action)


def has_message(action: ActionType) -> bool:
    return action in TEMPLATES
