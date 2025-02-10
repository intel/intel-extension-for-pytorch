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
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
    ) -> torch.Tensor:
        """
        This method extracts the image features from the vision backbone using the specified feature layer and
        selection strategy. This is custom to MAIRA-2 model since we want to use the `feature_maps` from the Dinov2Backbone
        class instead of the `hidden_states` which are used in the default implementation of `get_image_features`
        in LlavaForConditionalGeneration.
        The feature_maps returned by Dinov2Backbone are the hideen_states with a layernorm applied to them.
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.feature_maps[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features  # type: ignore[no-any-return]
