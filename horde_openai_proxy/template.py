import json
import os
from dataclasses import dataclass
from functools import cache
from typing import Optional

from transformers import PreTrainedTokenizerBase, AutoTokenizer

from .data import BASE_MODELS


@cache
def get_tokenizer(model: str) -> PreTrainedTokenizerBase:
    """
    Get the adjusted tokenizer for the model.
    :param model: Model name
    :return: Tokenizer
    """
    data = BASE_MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(data["model"], trust_remote_code=True)
    template_path = os.path.join(
        os.path.dirname(__file__),
        f"../chat_templates/chat_templates/{data['template']}.jinja",
    )
    template = open(template_path).read()
    template = template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = template
    return tokenizer


@dataclass
class GenerationConfig:
    system_prompt: Optional[str]
    stop_words: list[str]


@cache
def get_generation_config(model: str) -> GenerationConfig:
    """
    Get the generation config for the model, such as stop words.
    :param model: Model name
    :return: GenerationConfig
    """
    if BASE_MODELS[model]["config"]:
        config_path = os.path.join(
            os.path.dirname(__file__),
            f"../chat_templates/generation_configs/{BASE_MODELS[model]['config']}.json",
        )

        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"stop_str": None, "stop_token_ids": [], "system_prompt": None}

    stop_words: list[str] = []

    if config["stop_str"]:
        stop_words.append(str(config["stop_str"]))

    if config["stop_token_ids"]:
        tokenizer = get_tokenizer(model)
        for token_id in config["stop_token_ids"]:
            stop_words.append(tokenizer.decode(token_id))

    return GenerationConfig(
        system_prompt=config["system_prompt"],
        stop_words=stop_words,
    )


def apply_template(conversation: list, model: str) -> str:
    """
    Apply the chat template to the conversation
    :param conversation:  List of messages
    :param model: Model name
    :return: Prepared prompt
    """
    if "system" not in {m["role"] for m in conversation}:
        config = get_generation_config(model)
        if config.system_prompt:
            conversation = [
                {"role": "system", "content": config.system_prompt}
            ] + conversation

    return str(
        get_tokenizer(model).apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    )
