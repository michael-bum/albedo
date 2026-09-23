from __future__ import annotations

JUDGE_MODELS: tuple[str, ...] = ("z-ai/glm-5.2",)

JUDGE_PROVIDER_PINS: dict[str, dict[str, object]] = {
    model: {
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
        "order": ["streamlake", "baidu", "alibaba", "phala"],
    }
    for model in JUDGE_MODELS
}

JUDGE_LOGPROB_PROVIDER_PINS: dict[str, dict[str, object]] = {
    model: {
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
        "order": ["ambient", "alibaba"],
    }
    for model in JUDGE_MODELS
}

EVALUATOR_MODEL = "z-ai/glm-5.2"
EVALUATOR_PROVIDERS = "streamlake,baidu"
SOTA_MODELS = "z-ai/glm-5.2"
SIMULATION_MODEL = "deepseek/deepseek-v4.1-flash"
SIMULATION_PROVIDERS = "deepseek,fireworks,novita,streamlake"

JUDGE_PROVIDER_PINS[SIMULATION_MODEL] = {
    "allow_fallbacks": False,
    "order": ["deepseek", "fireworks", "novita", "streamlake"],
}
ENGY_MODELS = "z-ai/glm-5.2,deepseek/deepseek-v4-flash-0731"
