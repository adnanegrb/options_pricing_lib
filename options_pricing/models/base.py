import numpy as np
from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseType(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionParams:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = OptionType.CALL
    exercise: ExerciseType = ExerciseType.EUROPEAN
    q: float = 0.0

    def __post_init__(self):
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not isinstance(self.option_type, OptionType):
            self.option_type = OptionType(self.option_type)
        if not isinstance(self.exercise, ExerciseType):
            self.exercise = ExerciseType(self.exercise)

    @property
    def is_call(self):
        return self.option_type == OptionType.CALL

    @property
    def phi(self):
        return 1 if self.is_call else -1

    def intrinsic_value(self):
        return max(self.phi * (self.S - self.K), 0.0)
