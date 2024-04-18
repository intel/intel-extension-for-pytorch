"""This module provides predefined FP8 recipes."""

from __future__ import annotations
from enum import Enum
from typing import Literal, NamedTuple
from dataclasses import dataclass


class _FormatHelper(NamedTuple):
    """Stores max FP8 values for fprop and bprop a `Format`."""

    max_fwd: float
    max_bwd: float


class Format(Enum):
    """
    Supported FP8 formats.
    Values
    ------
    E4M3 :
          All FP8 tensors are in e4m3 format
    E5M2 :
          All FP8 tensors are in e5m2 format
    HYBRID :
            FP8 tensors in the forward pass are in e4m3 format,
            FP8 tensors in the backward pass are in e5m2 format
    """

    E4M3 = _FormatHelper(max_fwd=448, max_bwd=448)
    E5M2 = _FormatHelper(max_fwd=57344, max_bwd=57344)
    HYBRID = _FormatHelper(max_fwd=E4M3.max_fwd, max_bwd=E5M2.max_bwd)


@dataclass()
class DelayedScaling:
    margin: int = 0
    interval: int = 1
    fp8_format: Format = Format.HYBRID
    amax_history_len: int = 1024
    amax_compute_algo: Literal["max", "most_recent"] = "max"
