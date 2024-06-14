import torch
from torch.ao.quantization import (
    UniformQuantizationObserverBase,
    HistogramObserver,
    PerChannelMinMaxObserver,
)
import copy


class SmoothQuantActivationObserver(UniformQuantizationObserverBase):
    """
    For SmoothQuant, see https://arxiv.org/pdf/2211.10438.pdf
    Activation shape = T * IC (tokens * input channels)

    If smooth_quant_enabled is True (e.g. for nn.Linear module)
    1. Find max(|X_j|) for each IC
    2. Get max(|W_j|) for each IC in weight from the weight observer
    3. Calculate scaling factors for each IC by s_j = (max(|W_j|) ** (1 - alpha)) / (max(|X_j|) ** alpha)
        Note that factors for activation are reciprocals of that for weight
    4. Apply s_j to activation
    5. Find q-params per tensor and return

    If smooth_quant_enabled is False (i.e. for other ops, including functional linear),
      just act as a normal observer
    """

    def __init__(
        self,
        act_observer=None,
        act_ic_observer=None,
        smooth_quant_enabled=False,  # if false, act as a normal observer
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        alpha=0.5,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
    ) -> None:
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
        )
        self.weight_obs = None
        if act_ic_observer is None:
            self.ic_obs = PerChannelMinMaxObserver(
                ch_axis=-1,
                dtype=dtype,
                qscheme=torch.per_channel_affine,
                reduce_range=reduce_range,
                quant_min=quant_min,
                quant_max=quant_max,
                factory_kwargs=factory_kwargs,
                eps=eps,
            )
        else:
            self.ic_obs = act_ic_observer()
        if act_observer is None:
            self.act_obs = HistogramObserver(
                dtype=dtype,
                qscheme=qscheme,
                reduce_range=reduce_range,
                quant_min=quant_min,
                quant_max=quant_max,
                factory_kwargs=factory_kwargs,
                eps=eps,
            )
        else:
            self.act_obs = act_observer()
        # if smooth_quant_enabled is false, this observer acts as
        # a normal per-tensor observer
        self.smooth_quant_enabled = smooth_quant_enabled
        self.alpha = float(alpha)
        # Normally we don't use min_val or max_val here
        # They are for checks, like `_check_observer_has_run`
        self.min_val = self.act_obs.min_val
        self.max_val = self.act_obs.max_val
        # Dict of tensors. Keys are weight IDs. Factors are 1d tensors, not diagonal
        self.scaling_factors = {}

    def forward(self, x_orig):
        if not self.smooth_quant_enabled:
            return self.act_obs.forward(x_orig)
        # Run act_obs to indicate the observer has run
        self.act_obs.forward(x_orig)
        # Call per-channel observer on IC to find scaling factor
        return self.ic_obs.forward(x_orig)

    @torch.jit.export
    def calculate_qparams(self):
        if not self.smooth_quant_enabled:
            return self.act_obs.calculate_qparams()
        scales, zero_points = {}, {}
        for k in self.weight_obs.keys():
            # Get weight per IC min/max from weight observer
            wei_min_per_ic = self.weight_obs[k].min_val
            wei_max_per_ic = self.weight_obs[k].max_val
            act_min_per_ic = self.ic_obs.min_val
            act_max_per_ic = self.ic_obs.max_val
            x_abs_max_per_ic = (
                torch.max(torch.abs(act_min_per_ic), torch.abs(act_max_per_ic)) + 1e-6
            )
            w_abs_max_per_ic = (
                torch.max(torch.abs(wei_min_per_ic), torch.abs(wei_max_per_ic)) + 1e-6
            )
            # Note: activation's scaling factors are reciprocals of weight's
            scaling_factor = torch.pow(w_abs_max_per_ic, 1 - self.alpha) / torch.pow(
                x_abs_max_per_ic, self.alpha
            )
            self.scaling_factors.update({k: scaling_factor})
            # Apply scaling factors to each IC's min/max
            act_min_per_ic_new = act_min_per_ic * scaling_factor.reshape(
                act_min_per_ic.shape
            )
            act_max_per_ic_new = act_max_per_ic * scaling_factor.reshape(
                act_max_per_ic.shape
            )
            min_val_per_tensor = torch.min(act_min_per_ic_new)
            max_val_per_tensor = torch.max(act_max_per_ic_new)
            scale, zp = self._calculate_qparams(min_val_per_tensor, max_val_per_tensor)
            scales.update({k: scale})
            zero_points.update({k: zp})
        return scales, zero_points

    def get_scaling_factors(self):
        if not self.smooth_quant_enabled:
            return None
        return self.scaling_factors

    def extra_repr(self):
        return "smooth_quant_enabled={}, alpha={}".format(
            self.smooth_quant_enabled, self.alpha
        )


class SmoothQuantWeightObserver(UniformQuantizationObserverBase):
    """
    For SmoothQuant, see https://arxiv.org/pdf/2211.10438.pdf
    Weight shape = OC * IC (output channels * input channels)

    If smooth_quant_enabled is True (e.g. for nn.Linear module)
    1. Find max(|W_j|) for each IC
    2. Get max(|X_j|) for each IC in activation from the activation observer
    3. Calculate scaling factors for each IC by s_j = (max(|X_j|) ** alpha) / (max(|W_j|) ** (1 - alpha))
        Note that factors for weight are reciprocals of that for activation
    4. Apply s_j to weight
    5. Find q-params per OC and return

    If smooth_quant_enabled is False (i.e. for other ops, including functional linear),
      just act as a normal observer
    """

    # As a 1d tensor, not diagonal
    scaling_factors: torch.Tensor
    # Need to keep original weight to calculate q-param after applying scaling factors
    w_orig: torch.Tensor

    def __init__(
        self,
        wei_observer=None,
        wei_ic_observer=None,
        smooth_quant_enabled=False,  # if false, act as a normal observer
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        alpha=0.5,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
    ) -> None:
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
        )
        self.act_obs = None
        if wei_observer is None:
            self.oc_obs = PerChannelMinMaxObserver(
                ch_axis=0,
                dtype=dtype,
                qscheme=qscheme,
                reduce_range=reduce_range,
                quant_min=quant_min,
                quant_max=quant_max,
                factory_kwargs=factory_kwargs,
                eps=eps,
            )
        else:
            self.oc_obs = wei_observer()
        if wei_ic_observer is None:
            self.ic_obs = PerChannelMinMaxObserver(
                ch_axis=1,
                dtype=dtype,
                qscheme=torch.per_channel_affine,
                reduce_range=reduce_range,
                quant_min=quant_min,
                quant_max=quant_max,
                factory_kwargs=factory_kwargs,
                eps=eps,
            )
        else:
            self.ic_obs = wei_ic_observer()
        # if smooth_quant_enabled is false, this observer acts as
        # a normal observer
        self.smooth_quant_enabled = smooth_quant_enabled
        self.alpha = float(alpha)
        # Normally we don't use min_val or max_val here
        # They are for checks, like `_check_observer_has_run`
        self.min_val = self.oc_obs.min_val
        self.max_val = self.oc_obs.max_val

    def forward(self, x_orig):
        if not self.smooth_quant_enabled:
            return self.oc_obs.forward(x_orig)
        # Copy original weight to apply scaling factor
        self.w_orig = copy.deepcopy(x_orig)
        # Call per-channel observer on IC to find scaling factor
        return self.ic_obs.forward(x_orig)

    @torch.jit.export
    def calculate_qparams(self):
        if not self.smooth_quant_enabled:
            return self.oc_obs.calculate_qparams()
        # Get activation min/max per IC from activation observer
        act_min_per_ic = self.act_obs.min_val
        act_max_per_ic = self.act_obs.max_val
        wei_min_per_ic = self.ic_obs.min_val
        wei_max_per_ic = self.ic_obs.max_val
        w_abs_max_per_ic = (
            torch.max(torch.abs(wei_min_per_ic), torch.abs(wei_max_per_ic)) + 1e-6
        )
        x_abs_max_per_ic = (
            torch.max(torch.abs(act_min_per_ic), torch.abs(act_max_per_ic)) + 1e-6
        )
        # Note: weight's scaling factors are reciprocals of activation's
        self.scaling_factors = torch.pow(x_abs_max_per_ic, self.alpha) / torch.pow(
            w_abs_max_per_ic, 1 - self.alpha
        )

        # Apply scaling factors to original weight
        # w.shape = [OC, IC], len(scaling_factors) = IC
        w_new = torch.mul(self.w_orig, self.scaling_factors)
        # Run per-channel observer on new weight and return q-params
        self.oc_obs.reset_min_max_vals()
        self.oc_obs.forward(w_new)
        return self.oc_obs.calculate_qparams()

    def get_scaling_factors(self):
        if not self.smooth_quant_enabled:
            return None
        return self.scaling_factors

    def extra_repr(self):
        return "smooth_quant_enabled={}, alpha={}".format(
            self.smooth_quant_enabled, self.alpha
        )
