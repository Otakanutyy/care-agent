"""Evaluation framework: adversarial merchant simulator, scenario runner, metrics, and judge."""

from care_agent.eval.runner import (
    SCENARIO_DIR,
    RunResult,
    Scenario,
    load_scenario,
    load_scenarios,
    run_matrix,
    run_once,
)
from care_agent.eval.simulator import (
    PERSONA_DIR,
    MerchantSimulator,
    Persona,
    ScriptedMerchantSimulator,
    build_simulator,
    load_persona,
    load_personas,
    render_history,
)

__all__ = [
    "MerchantSimulator",
    "PERSONA_DIR",
    "Persona",
    "RunResult",
    "SCENARIO_DIR",
    "Scenario",
    "ScriptedMerchantSimulator",
    "build_simulator",
    "load_persona",
    "load_personas",
    "load_scenario",
    "load_scenarios",
    "render_history",
    "run_matrix",
    "run_once",
]
