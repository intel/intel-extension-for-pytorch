from enum import IntEnum


class QuantMethod(IntEnum):
    GPTQ_GEMM = 0
    AWQ_GEMM = 1


class QuantDtype(IntEnum):
    INT4 = 0
    FP8_E5M2 = 1
    FP8_E4M3FN = 2
