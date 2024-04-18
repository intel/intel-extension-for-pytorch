import torch
import io
from contextlib import contextmanager
import torch.nn.functional as F
import intel_extension_for_pytorch._isa_help as ipex
from .fp8 import (
    is_fp8_enabled,
    is_fp8_calibration,
    get_default_fp8_recipe,
    get_fp8_recipe,
    get_fp8_device_type,
    amax_and_scale_update,
)


class Fp8BaseModule(torch.nn.Module):
    """Base FP8 module."""

    def __init__(self, **kwargs) -> None:
        super(Fp8BaseModule, self).__init__(**kwargs)
        self.fp8 = False
        self.fp8_meta = {}
        self.fp8_initialized = False
        self.fp8_calibration = False
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

        if self.fp8_meta_tensors_initialized:
            # Handle changed amax history size.
            curr_len = self.fp8_meta[fp8_meta_tensor_key].amax_history.shape[0]
            need_len = self.fp8_meta["recipe"].amax_history_len
            if need_len < curr_len:
                self.fp8_meta[fp8_meta_tensor_key].amax_history = (
                    self.fp8_meta[fp8_meta_tensor_key]
                    .amax_history[: self.fp8_meta["recipe"].amax_history_len]
                    .clone()
                )
            elif need_len > curr_len:
                extra_rows = need_len - curr_len
                self.fp8_meta[fp8_meta_tensor_key].amax_history = F.pad(
                    self.fp8_meta[fp8_meta_tensor_key].amax_history,
                    pad=(0, 0, 0, extra_rows),
                )
            return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 3 if fwd else self.fp8_meta["num_gemms"] * 2
        )

        self.fp8_meta[fp8_meta_tensor_key] = ipex.FP8TensorMeta()
        self.fp8_meta[fp8_meta_tensor_key].scale = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device=get_fp8_device_type()
        )
        self.fp8_meta[fp8_meta_tensor_key].scale_inv = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device=get_fp8_device_type()
        )
        self.fp8_meta[fp8_meta_tensor_key].amax_history = torch.zeros(
            [self.fp8_meta["recipe"].amax_history_len, num_fp8_tensors],
            dtype=torch.float32,
            device=get_fp8_device_type(),
        )

    def init_fp8_meta_tensors(self) -> None:
        """Init scales and amaxes."""
        self.set_meta_tensor(True)
        self.set_meta_tensor(False)
        self.fp8_meta_tensors_initialized = True

    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        self.fp8 = is_fp8_enabled()
        self.fp8_calibration = is_fp8_calibration()

        if self.fp8 or self.fp8_calibration:
            # FP8 init has already been run and recipe is the same, don't do anything.
            if self.fp8_initialized and get_fp8_recipe() == self.fp8_meta["recipe"]:
                return

            # Set FP8, recipe, and other FP8 metadata
            self.fp8_meta["recipe"] = get_fp8_recipe()
            self.fp8_meta["num_gemms"] = num_gemms

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta[
                "recipe"
            ].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta[
                "recipe"
            ].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            self.init_fp8_meta_tensors()
            self.fp8_initialized = True
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

    @contextmanager
    def prepare_forward(self, num_gemms=1):
        """Checks and prep for FWD."""
        self.fp8_init(num_gemms)
        amax_and_scale_update(self.fp8_meta, fwd_update=True)
        yield

    def get_extra_state(self) -> torch.Tensor:
        """Save before checkpointing."""
        state = None

        fp8_checkpoint = self.fp8 or self.fp8_calibration

        if fp8_checkpoint:
            state = {}
            state["scale_fwd"] = self.fp8_meta["scaling_fwd"].scale
            state["scale_inv_fwd"] = self.fp8_meta["scaling_fwd"].scale_inv
            state["amax_history_fwd"] = self.fp8_meta["scaling_fwd"].amax_history
            state["scale_bwd"] = self.fp8_meta["scaling_bwd"].scale
            state["scale_inv_bwd"] = self.fp8_meta["scaling_bwd"].scale_inv
            state["amax_history_bwd"] = self.fp8_meta["scaling_bwd"].amax_history

            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str, list)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        state_serialized = io.BytesIO()
        torch.save(state, state_serialized)

        return state_serialized

    def set_extra_state(self, state: torch.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        if isinstance(state, io.BytesIO):
            state.seek(0)
            state = torch.load(state, map_location=get_fp8_device_type())
        else:
            raise RuntimeError("Unsupported checkpoint format.")

        if state is None:
            return

        # Load extra items.
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"].amax_history_len = state["amax_history_fwd"].shape[0]

        # Initialize before loading.
        self.init_fp8_meta_tensors()
        self.fp8_initialized = True
        self.fp8_meta["scaling_fwd"].scale.copy_(state["scale_fwd"])
        self.fp8_meta["scaling_fwd"].amax_history.copy_(state["amax_history_fwd"])
        self.fp8_meta["scaling_bwd"].scale.copy_(state["scale_bwd"])
        self.fp8_meta["scaling_bwd"].amax_history.copy_(state["amax_history_bwd"])
        self.fp8_meta["scaling_fwd"].scale_inv.copy_(state["scale_inv_fwd"])
        self.fp8_meta["scaling_bwd"].scale_inv.copy_(state["scale_inv_bwd"])


@contextmanager
def prepare_backward(fp8_meta, num_gemms=1):
    """Checks and prep for BWD."""
    amax_and_scale_update(fp8_meta, fwd_update=False)
    yield
