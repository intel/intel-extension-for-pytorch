"""FP8 utilies for IPEX"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple

import torch
from .recipe import DelayedScaling, Format
import intel_extension_for_pytorch._isa_help as ipex

dist_group_type = torch.distributed.ProcessGroup


def get_default_fp8_recipe() -> DelayedScaling:
    return DelayedScaling()


def get_fp8_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> str:
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return ipex.Float8Format.kFloat8_E4M3  # "E4M3"
    return ipex.Float8Format.kFloat8_E5M2  # "E5M2"


class FP8GlobalStateManager:
    """Class to keep track of and manipulate the global
    FP8 state at different stages of execution.
    """

    FP8_ENABLED = False
    FP8_CALIBRATION = False
    FP8_RECIPE = None
    FP8_AUTOCAST_DEPTH = 0
    FP8_DEVICE = "xpu"

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        cls.FP8_ENABLED = False
        cls.FP8_CALIBRATION = False
        cls.FP8_RECIPE = None
        cls.FP8_AUTOCAST_DEPTH = 0
        cls.FP8_DEVICE = "xpu"

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.FP8_ENABLED

    @classmethod
    def is_fp8_calibration(cls) -> bool:
        """Is FP8 calibration"""
        return cls.FP8_CALIBRATION

    @classmethod
    def get_fp8_recipe(cls) -> DelayedScaling:
        """Return the fp8 recipe"""
        return cls.FP8_RECIPE

    @classmethod
    def set_fp8_device_type(cls, device) -> None:
        cls.FP8_DEVICE = device

    @classmethod
    def get_fp8_device_type(cls):
        return cls.FP8_DEVICE

    @classmethod
    def get_fp8_autocast_state(cls) -> Tuple[bool, bool, DelayedScaling]:
        """FP8 autocast state getter"""
        return (cls.FP8_ENABLED, cls.FP8_CALIBRATION, cls.FP8_RECIPE)

    @classmethod
    def set_fp8_autocast_state(
        cls, fp8_state: Tuple[bool, bool, DelayedScaling]
    ) -> None:
        """FP8 autocast state setter"""
        (cls.FP8_ENABLED, cls.FP8_CALIBRATION, cls.FP8_RECIPE)

    @classmethod
    def fp8_autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[DelayedScaling] = None,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""
        cls.FP8_ENABLED = enabled
        cls.FP8_CALIBRATION = calibrating
        cls.FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        cls.FP8_AUTOCAST_DEPTH += 1

    @classmethod
    def fp8_autocast_exit(cls):
        """Set state and tracking variables for exit from FP8 region."""
        cls.FP8_AUTOCAST_DEPTH -= 1


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    calibrating: bool = False,
    fp8_recipe: Optional[DelayedScaling] = None,
    fp8_group: Optional[dist_group_type] = None,
    device: str = "xpu",
) -> None:
    """
    Context manager for FP8 usage.

    .. code-block:: python

        with fp8_autocast(enabled=True):
            out = model(inp)

    Parameters
    ----------
    enabled: bool, default = `True`
             whether or not to enable fp8
    calibrating: bool, default = `False`
                 calibration mode allows collecting statistics such as amax and scale
                 data of fp8 tensors even when executing without fp8 enabled.
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """
    try:
        fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
        assert (
            fp8_group is None
        ), "Don't support fp8_group, will support when enable distributed running."
        FP8GlobalStateManager.fp8_autocast_enter(
            enabled=enabled, calibrating=calibrating, fp8_recipe=fp8_recipe
        )
        FP8GlobalStateManager.set_fp8_device_type(device)
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
        FP8GlobalStateManager.fp8_autocast_exit()


def update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        amax_history = torch.roll(amax_history, -1, 0)
    amax_history[0].fill_(0.0)
    return amax_history


def _default_get_amax(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = torch.max(amax_history, dim=0).values
    else:
        amax = amax_history[0]

    amax_history = update_amax_history(amax_history)
    return amax_history, amax


def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""
    exp = torch.floor(torch.log2(fp8_max / amax)) - margin
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(exp < 0, 1 / sf, sf)
    return sf


def default_amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    amax_compute_algo: str,
    recipe: DelayedScaling,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Amax to scale conversion."""

    # Get amax from history.
    if recipe.scaling_factor_compute_algo is None:
        amax_history, amax = _default_get_amax(
            amax_history,
            amax_compute_algo,
        )

        # Calculate new scaling factor.
        scale = _default_sf_compute(
            amax,
            scale,
            fp8_max,
            margin,
        )

        return amax_history, scale, 1.0 / scale
    else:
        return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def _compute_amax(
    amax_history: torch.Tensor,
    recipe: DelayedScaling,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""

    if callable(recipe.amax_compute_algo):
        amax = recipe.amax_compute_algo(amax_history)
        amax_history = _update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax(
        amax_history,
        recipe.amax_compute_algo,
    )


def _compute_scaling_factor(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> torch.Tensor:
    """Convert amax to scaling factor."""

    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(
            amax,
            scale,
            fp8_max,
            recipe.margin,
        )
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def _compute_scaling_factor_inverse(
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
    non_weight_mask: torch.Tensor,
    update_weight_scale_inv: bool,
) -> torch.Tensor:
    """Compute inverse of scaling factor."""
    if update_weight_scale_inv:
        return 1.0 / scale
    return torch.where(non_weight_mask, 1.0 / scale, scale_inv)


def amax_and_scale_update(
    fp8_meta: Dict[str, Any],
    fwd_update: bool,
    update_weight_scale_inv: bool = True,
) -> None:
    """Updates fp8 amaxes/scales for fwd | bwd."""
    amax_compute = fp8_meta["recipe"].amax_compute_algo
    sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo
    fp8_meta_tensor_key = "scaling_fwd" if fwd_update else "scaling_bwd"
    fp8_max_key = "fp8_max_fwd" if fwd_update else "fp8_max_bwd"

    if not callable(amax_compute) and sf_compute is None:
        (
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_meta_tensor_key].scale_inv,
        ) = default_amax_and_scale_update(
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_max_key],
            fp8_meta["recipe"].margin,
            fp8_meta["recipe"].amax_compute_algo,
            fp8_meta["recipe"],
        )
    else:
        fp8_meta[fp8_meta_tensor_key].amax_history, amax = _compute_amax(
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta["recipe"],
        )
        fp8_meta[fp8_meta_tensor_key].scale = _compute_scaling_factor(
            amax,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_max_key],
            fp8_meta["recipe"],
        )
        fp8_meta[fp8_meta_tensor_key].scale_inv = _compute_scaling_factor_inverse(
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_meta_tensor_key].scale_inv,
            fp8_meta[fp8_meta_tensor_key + "_non_weight_mask"],
            update_weight_scale_inv,
        )
