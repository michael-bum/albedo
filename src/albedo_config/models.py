from __future__ import annotations

GLM_5_3_FLASH = "z-ai/glm-5.3-flash"
GLM_5_2 = "z-ai/glm-5.2"

JUDGE_MODELS: tuple[str, ...] = (GLM_5_3_FLASH,)
JUDGE_FALLBACK_MODELS: dict[str, str] = {GLM_5_3_FLASH: GLM_5_2}
MODEL_REASONING: dict[str, dict[str, object]] = {GLM_5_3_FLASH: {"effort": "low", "exclude": True}}

# OpenRouter providers in the order to try them, flagged by whether their top-20 logprobs line up
# with the sampled token. Every purpose walks the whole roster; a judge call only the aligned ones.
PROVIDERS: dict[str, dict[str, bool]] = {
    GLM_5_3_FLASH: {
        "friendli": False,
        "together": False,
        "parasail": True,
        "inference-net/fp4": False,
        "phala/nvfp4": False,
        "decart/fp4": False,
        "reka": True,
        "digitalocean": True,
    },
    GLM_5_2: {
        "streamlake": False,
        "baidu": False,
        "alibaba": True,
        "phala": False,
        "digitalocean": True,
    },
}

EVALUATOR_MODEL = GLM_5_3_FLASH
EVALUATOR_PROVIDERS = ",".join(PROVIDERS[GLM_5_3_FLASH])
SOTA_MODELS = GLM_5_3_FLASH
SIMULATION_MODEL = "deepseek/deepseek-v4.1-flash"
SIMULATION_PROVIDERS = "streamlake,atlas-cloud,deepinfra,deepseek"
ENGY_MODELS = "z-ai/glm-5.3-flash,deepseek/deepseek-v4-flash-0731"


def _pins(model: str, logprobs: bool = False) -> dict[str, object]:
    """The roster as a provider block; the quantization is part of the endpoint slug."""
    order = [p for p, aligned in PROVIDERS[model].items() if aligned or not logprobs]
    return {"allow_fallbacks": False, "order": order}


JUDGE_PROVIDER_PINS: dict[str, dict[str, object]] = {
    **{model: _pins(model) for model in PROVIDERS},
    SIMULATION_MODEL: {"allow_fallbacks": False, "order": SIMULATION_PROVIDERS.split(",")},
}
JUDGE_LOGPROB_PROVIDER_PINS: dict[str, dict[str, object]] = {
    model: _pins(model, logprobs=True) for model in PROVIDERS
}
