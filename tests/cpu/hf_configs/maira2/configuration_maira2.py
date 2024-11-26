#  Copyright 2024 Microsoft. All rights reserved.
#  Licensed under the MSRLA License. See LICENSE in the repo root for license information.


from typing import Any

from transformers import LlavaConfig


class Maira2Config(LlavaConfig):
    """
    This is the configuration class to store the configuration of a `Maira2ForConditionalGeneration` model. It is
    used to instantiate a MAIRA-2 model according to the specified arguments, defining the model architecture.

    It inherits from `LlavaConfig`. In addition to the inherited attributes, it adds the
    ability to customize the multimodal projector through following attributes:

    Args:
        projector_n_layers (`int`, *optional*, defaults to 4):
            Number of layers in the multimodal projector.
    """

    model_type = "maira2"

    def __init__(
        self,
        projector_n_layers: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = self.text_config.hidden_size
        self.projector_n_layers = projector_n_layers
