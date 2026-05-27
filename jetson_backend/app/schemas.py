from enum import Enum
from pydantic import BaseModel


class Mode(str, Enum):
    kids = "kids"
    young_adult = "young_adult"
    adult = "adult"


class ModeRequest(BaseModel):
    mode: Mode


class VoiceCommandRequest(BaseModel):
    command: str


class RobotActionRequest(BaseModel):
    action: str


class LLMChatRequest(BaseModel):
    message: str


class EyeRequest(BaseModel):
    expression: str  # normal | excited | disappointed | blink | sleeping


class LedRequest(BaseModel):
    color: str  # red | green | yellow | all_on | off