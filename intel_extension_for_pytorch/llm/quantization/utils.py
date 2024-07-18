from enum import IntEnum


class QuantMethod(IntEnum):
    GPTQ_GEMM = 0
    AWQ_GEMM = 1


class QuantDtype(IntEnum):
    INT4 = 0
