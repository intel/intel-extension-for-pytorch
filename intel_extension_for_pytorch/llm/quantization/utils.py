from enum import IntEnum


class QuantMethod(IntEnum):
    GPTQ_GEMM = 0
    AWQ_GEMM = 1


class QuantDtype(IntEnum):
    INT4 = 0
    FP8_E5M2 = 1
    FP8_E4M3FN = 2


class XPUWoqActQuantMode(IntEnum):
    """
    Float activation: [M, K].
    UNQUANT_A: no quantization for activation, apply w4a16 gemm.

    For symmetric quantization, activaiton is quantized to int8 with zero point 0; otherwise uint8
    with non-zero zp.

    -----------------------------------------
    scheme            |  scale & zp shape
    -----------------------------------------
    per-tensor        |   [1]
    per-M / per-token |   [M, 1]
    per-K-block       |   [K/k_block_size, 1]
    per-M-K-block     |   [M, K/k_block_size]
    """

    UNQUANT_A = -1
    QUANT_A_PER_TENSOR = 0
    QUANT_A_PER_TENSOR_SYM = 1
    QUANT_A_PER_M = 2
    QUANT_A_PER_M_SYM = 3
    QUANT_A_PER_K_BLOCK = 4
    QUANT_A_PER_K_BLOCK_SYM = 5
    QUANT_A_PER_M_K_BLOCK = 6
    QUANT_A_PER_M_K_BLOCK_SYM = 7


# definitions from vLLM
class WoqActQuantMode(IntEnum):
    NONE = -1
    PER_TENSOR = 0
    PER_IC_BLOCK = 1  # IC = Input Channel
    PER_BATCH = 2
    PER_BATCH_IC_BLOCK = 3
    PER_TENSOR_SYM = 4
    PER_IC_BLOCK_SYM = 5
    PER_BATCH_SYM = 6
    PER_BATCH_IC_BLOCK_SYM = 7


XPU_UNSUPPORTED_ACT_QUANT_MODES = [
    WoqActQuantMode.PER_IC_BLOCK,
    WoqActQuantMode.PER_IC_BLOCK_SYM,
    WoqActQuantMode.PER_BATCH_IC_BLOCK,
    WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
]

VLLM_ACT_QUANT_MODE_TO_XPU = {
    WoqActQuantMode.NONE: XPUWoqActQuantMode.UNQUANT_A,
    WoqActQuantMode.PER_TENSOR: XPUWoqActQuantMode.QUANT_A_PER_TENSOR,
    WoqActQuantMode.PER_TENSOR_SYM: XPUWoqActQuantMode.QUANT_A_PER_TENSOR_SYM,
    WoqActQuantMode.PER_BATCH: XPUWoqActQuantMode.QUANT_A_PER_M,
    WoqActQuantMode.PER_BATCH_SYM: XPUWoqActQuantMode.QUANT_A_PER_M_SYM,
    WoqActQuantMode.PER_IC_BLOCK: XPUWoqActQuantMode.QUANT_A_PER_K_BLOCK,
    WoqActQuantMode.PER_IC_BLOCK_SYM: XPUWoqActQuantMode.QUANT_A_PER_K_BLOCK_SYM,
    WoqActQuantMode.PER_BATCH_IC_BLOCK: XPUWoqActQuantMode.QUANT_A_PER_M_K_BLOCK,
    WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM: XPUWoqActQuantMode.QUANT_A_PER_M_K_BLOCK_SYM,
}
