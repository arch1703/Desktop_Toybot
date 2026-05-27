from enum import Enum
from pydantic import BaseModel


class Mode(str, Enum):
    kids = "kids"
    young_adult = "young_adult"
    adult = "adult"


class DeviceModeRequest(BaseModel):
    mode: Mode


class DeviceActionRequest(BaseModel):
    action: str


class EyeExpressionRequest(BaseModel):
    expression: str   # normal | excited | disappointed | blink | sleeping


class LedColorRequest(BaseModel):
    color: str        # red | green | yellow | all_on | off


class AudioRecordRequest(BaseModel):
    duration: int = 5   # seconds


class AudioSpeakRequest(BaseModel):
    text: str