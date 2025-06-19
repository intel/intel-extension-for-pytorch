#  Copyright 2024 Microsoft. All rights reserved.
#  Licensed under the MSRLA License. See LICENSE in the repo root for license information.


import torch
from torch.nn import Linear, Module, Sequential
from transformers import (
    AutoBackbone,
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    LlavaPreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.utils import check_min_version
from typing import List, Union

from .configuration_maira2 import Maira2Config


class Maira2MultiModalProjector(Module):
    """
    This class implements the multimodal projector for MAIRA-2 model. It projects the image features to the text
    hidden size via a series of linear layers (4 layers in MAIRA-2).
    """

    def __init__(self, config: Maira2Config):
        super().__init__()

        n_layers = config.projector_n_layers
        if n_layers < 1:
            raise ValueError(f"Number of layers should be at least 1, got {n_layers=}")
        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size
        _layers = [Linear(vision_hidden_size, text_hidden_size, bias=True)]
        for _ in range(n_layers - 1):
            _layers.append(ACT2FN[config.projector_hidden_act])
            _layers.append(Linear(text_hidden_size, text_hidden_size, bias=True))

        self.layers = Sequential(*_layers)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        hidden_states = self.layers(image_features)
        return hidden_states  # type: ignore[no-any-return]


class Maira2ForConditionalGeneration(LlavaForConditionalGeneration):
    """
    This model implements the multimodal model MAIRA-2. It consists of a vision backbone, a multimodal projector, and a
    language model. The model can be used for grounded and ungrounded report generation tasks as well as phrase grounding.
    This class inherits from `LlavaForConditionalGeneration`, defining a custom multimodal projector and changing image
    feature selection.
    """

    config_class = Maira2Config

    def __init__(self, config: Maira2Config) -> None:

        # Check transformers version is at least 4.46.0.dev0  otherwise the model fails
        # silently since get_image_features is not called in the forward pass
        check_min_version("4.46.0.dev0")

        super(LlavaPreTrainedModel, self).__init__(config)
        self.vision_tower = AutoBackbone.from_config(config.vision_config)

        self.multi_modal_projector = Maira2MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation=config._attn_implementation,
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(
            pixel_values, output_hidden_states=True, **kwargs
        )

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features
