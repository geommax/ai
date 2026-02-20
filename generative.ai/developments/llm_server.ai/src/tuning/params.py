"""TuningParams — generation hyper-parameter dataclass."""

from dataclasses import dataclass


@dataclass
class TuningParams:
    """Generation tuning parameters."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    repetition_penalty: float = 1.1
    do_sample: bool = True
