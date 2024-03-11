import torch._decomp as decomp

from ..utils._logger import logger

decomposition_overrides = {}


def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decomposition_overrides:
            logger.warning(f"duplicate decomp: {ops}")
    return decomp.register_decomposition(ops, decomposition_overrides)


# Add custom decompositions here with `register_decomposition` decorator


def get_decompositions():
    from torch._inductor.decomposition import select_decomp_table

    return {**select_decomp_table(), **decomposition_overrides}
