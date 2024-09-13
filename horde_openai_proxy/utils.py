from typing import List

from .model import get_models, Model


def filter_models(
    names: set[str],
    clean_names: set[str],
    base_models: set[str],
    backends: set[str],
    quant: set[str],
    min_size: float = 0,
    max_size: float = -1,
) -> List[Model]:
    """
    Returns a filtered list of models.
    :param names: Raw names as used in the API
    :param clean_names: Clean names, without size, quant, backend, organization, ...
    :param base_models: Base models
    :param backends:
    :param quant:
    :param min_size:
    :param max_size:
    :return:
    """
    filtered_models = []
    for model in get_models().values():
        if names and model.name not in names:
            continue
        if clean_names and model.clean_name not in clean_names:
            continue
        if base_models and model.base_model not in base_models:
            continue
        if backends and model.backend not in backends:
            continue
        if quant and model.quant not in quant:
            continue
        if min_size > model.size:
            continue
        if max_size != -1 and model.size > max_size:
            continue
        filtered_models.append(model)
    return filtered_models
