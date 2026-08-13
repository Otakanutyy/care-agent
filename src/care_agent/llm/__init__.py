"""The two LLM edges: a classifier that reads text in, and a generator that writes text out.
Neither can originate an action — every decision comes from the policy engine."""

from care_agent.llm.classifier import (
    CLASSIFIER_SCHEMA,
    ClassifierError,
    classify,
    detect_human_request,
    detect_language,
)
from care_agent.llm.client import (
    CLASSIFIER_MODEL,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    AnthropicClient,
    LLMClient,
    LLMError,
    MockLLMClient,
)
from care_agent.llm.generator import GeneratedReply, GeneratorError, generate
from care_agent.llm.templates import LANGUAGES, TEMPLATES, has_message, spec_for

__all__ = [
    "AnthropicClient",
    "CLASSIFIER_MODEL",
    "CLASSIFIER_SCHEMA",
    "ClassifierError",
    "GENERATOR_MODEL",
    "GeneratedReply",
    "GeneratorError",
    "JUDGE_MODEL",
    "LANGUAGES",
    "LLMClient",
    "LLMError",
    "MockLLMClient",
    "TEMPLATES",
    "classify",
    "detect_human_request",
    "detect_language",
    "generate",
    "has_message",
    "spec_for",
]
