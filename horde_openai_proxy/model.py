from dataclasses import dataclass
from typing import Optional

from cachetools import TTLCache, cached

from .data import MODEL_SIZES, MODEL_TO_BASE_MODEL
from .horde import get_horde_models

QUANTS = {
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_0",
    "q4_k_s",
    "q4_k_m",
    "q5_0",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
}

IGNORED_WORDS = {"GGUF"}


def estimate_clean_name(name: str) -> str:
    name = name.split("/")[-1]
    filtered = []
    for part in name.split("-"):
        if len(part) <= 4 and part.lower().endswith("b"):
            continue
        if part.lower() in QUANTS:
            continue
        if part.lower() in IGNORED_WORDS:
            continue
        filtered.append(part)
    return "-".join(filtered)


def estimate_quant(name: str) -> str:
    name = name.lower()
    for q in QUANTS:
        if q in name:
            return q
    return "full"


def estimate_size(name: str) -> float:
    short_name = name.split("/")[-1]
    if name in MODEL_SIZES:
        return MODEL_SIZES[name]
    if short_name in MODEL_SIZES:
        return MODEL_SIZES[short_name]
    for s in name.lower().split("-"):
        if s.endswith("b"):
            try:
                return float(s[:-1])
            except ValueError:
                pass
    print(f"Unknown size: {name}")
    return 0


def guess_model_name(name: str) -> Optional[str]:
    name = name.lower()
    if name in MODEL_TO_BASE_MODEL:
        return MODEL_TO_BASE_MODEL[name]
    for k, v in MODEL_TO_BASE_MODEL.items():
        if k in name:
            return v
    return None


@dataclass
class Model:
    name: str
    clean_name: str
    base_model: str
    backend: str
    quant: str
    size: float


@cached(TTLCache(maxsize=1, ttl=3600))
def get_models() -> dict[str, Model]:
    """
    Get all models from the Horde API, with estimated sizes.
    """
    models = {}
    for model in get_horde_models():
        name = model["name"]
        if "/" in name:
            backend = name.split("/", 1)[0].strip()
            clean_name = estimate_clean_name(name).strip()
            base_model = guess_model_name(clean_name)

            if base_model is None:
                print(f"Unknown model: {clean_name} ({name}), ignoring.")
            else:
                models[name] = Model(
                    name=name,
                    clean_name=clean_name,
                    base_model=base_model,
                    backend=backend,
                    quant=estimate_quant(name),
                    size=estimate_size(name),
                )
    return models
