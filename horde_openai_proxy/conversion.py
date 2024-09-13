from .model import get_models
from .template import apply_template, get_generation_config
from .types import (
    ChatCompletionRequest,
    HordeRequest,
    ModelGenerationInput,
)


def openai_to_horde(
    request: ChatCompletionRequest,
    max_context_length: int = 2048,
) -> HordeRequest:
    """
    Convert an OpenAI request to a Horde request.

    :param request: The OpenAI request
    :param max_context_length: The maximum context length (not applicable to OpenAI and thus a constant)
    :return: The Horde request
    """
    model_names = [m.strip() for m in request.model.split(",")]
    models = get_models()
    primary_model = model_names[0]
    if primary_model not in models:
        raise ValueError(f"Model {primary_model} not known!")
    base_model = models[primary_model].base_model

    return HordeRequest(
        prompt=apply_template(request.messages, base_model),
        models=model_names,
        timeout=300 if request.timeout is None else int(request.timeout),
        params=ModelGenerationInput(
            max_context_length=max_context_length,
            max_length=request.max_tokens,
            n=request.n,
            rep_pen=request.frequency_penalty,
            stop_sequence=([] if request.stop is None else request.stop)
            + get_generation_config(base_model).stop_words,
            temperature=request.temperature,
            top_p=request.top_p,
        ),
    )


def horde_to_openai(request: HordeRequest) -> ChatCompletionRequest:
    """
    Convert a Horde request to an OpenAI request.

    :param request: The Horde request
    :return: The OpenAI request
    """
    params = request.params
    if params is None:
        raise ValueError("Request params are required")

    return ChatCompletionRequest(
        messages=[
            {
                "content": request.prompt,
                "role": "user",
            }
        ],
        model=request.models[0],
        frequency_penalty=params.rep_pen,
        presence_penalty=None,
        max_tokens=params.max_length,
        n=params.n,
        stop=[] if params.stop_sequence is None else params.stop_sequence,
        temperature=params.temperature,
        top_p=params.top_p,
        timeout=request.timeout,
    )