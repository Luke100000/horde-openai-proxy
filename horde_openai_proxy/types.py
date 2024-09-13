from typing import Optional, Union, List

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}

    messages: list[dict]
    model: str
    frequency_penalty: Optional[float] = Field(None)
    presence_penalty: Optional[float] = Field(None)
    max_tokens: Optional[int] = Field(None)
    n: Optional[int] = Field(None)
    stop: List[str] = Field(None)
    temperature: Optional[float] = Field(None)
    top_p: Optional[float] = Field(None)
    timeout: Union[float, None] = Field(None)


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list[dict]
    created: int
    model: str
    usage: dict


class ModelGenerationInput(BaseModel):
    max_context_length: int
    max_length: Optional[int]
    n: Optional[int]
    rep_pen: Optional[float]
    stop_sequence: List[str]
    temperature: Optional[float]
    top_p: Optional[float]


class HordeRequest(BaseModel):
    prompt: str = Field(...)
    models: List[str] = Field(...)
    timeout: int = Field((60 * 20) - 30)
    params: ModelGenerationInput


class TextGeneration(BaseModel):
    uuid: str
    model: str
    text: str
    kudos: int
