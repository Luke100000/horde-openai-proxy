import re
from typing import List

from . import ModelGenerationInput
from .data import BASE_MODELS
from .model import get_models, Model
from .template import get_tokenizer


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


def trim_incomplete_sentence(txt: str) -> str:
    last_punctuation = max(txt.rfind("."), txt.rfind("!"), txt.rfind("?"))

    # Is this the end of a quote?
    if last_punctuation < len(txt) - 1:
        if txt[last_punctuation + 1] == '"':
            last_punctuation = last_punctuation + 1

    if last_punctuation >= 0:
        txt = txt[: last_punctuation + 1]

    return txt


def apply_kobold_formatting(
    text: str,
    frmtadsnsp: bool = True,
    frmtrmblln: bool = True,
    frmtrmspch: bool = True,
    frmttriminc: bool = True,
    singleline: bool = True,
) -> str:
    """
    Apply KoboldAI formatting to the text.
    :param text: The text to format
    :param frmtadsnsp: Adds a leading space to your input if there is no trailing whitespace at the end of the previous action.
    :param frmtrmblln: Replaces all occurrences of two or more consecutive newlines in the output with one newline.
    :param frmtrmspch: Removes #/@%}{+=~|^<> from the output.
    :param frmttriminc: Removes some characters from the end of the output such that the output doesn't end in the middle of a sentence.
        If the output is less than one sentence long, does nothing.
    :param singleline: Removes everything after the first line of the output, including the newline.
    :return: The formatted text
    """
    if frmtadsnsp:
        pass
    if frmtrmblln:
        text = text.replace("\n\n", "\n")
    if frmtrmspch:
        text = re.sub(r"[/@%<>{}+=~|^]", "", text)
    if frmttriminc:
        text = trim_incomplete_sentence(text)
    if singleline:
        text = text.lstrip().split("\n")[0]
    return text


def apply_kobold_formatting_from_payload(
    text: str, payload: ModelGenerationInput
) -> str:
    """Apply KoboldAI formatting to the text using the payload.
    :param text: The text to format
    :param payload: The payload containing the formatting options
    :return: The formatted text
    """
    return apply_kobold_formatting(
        text,
        payload.frmtadsnsp,
        payload.frmtrmblln,
        payload.frmtrmspch,
        payload.frmttriminc,
        payload.singleline,
    )


def check_available_base_models():
    """Call to get a list of unavailable base models."""
    for base_model in BASE_MODELS:
        try:
            get_tokenizer(base_model)
        except Exception as e:
            url = BASE_MODELS[base_model]["model"]
            print(f"Model {base_model} at {url} not available: {e}")
