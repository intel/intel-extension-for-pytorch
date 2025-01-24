from test_llm_common_utils import TestIpexLLMOptimizeBase
from test_llm_common_utils import (
    curpath,
    common_params,
)
import transformers
import pytest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    parametrize,
)


class TestFalconOptimize(TestIpexLLMOptimizeBase):
    model_class = transformers.models.falcon.modeling_falcon.FalconForCausalLM
    model_config = {"config_path": f"{curpath}/hf_configs/falcon"}

    @parametrize("use_static_cache", common_params["use_static_cache"])
    @parametrize("num_beams", common_params["num_beams"])
    @parametrize("input_tokens_length", common_params["input_tokens_length"])
    @parametrize("max_new_tokens", common_params["max_new_tokens"])
    def test_falcon_model_generate(self, **params):
        if params["use_static_cache"] == False and params["input_tokens_length"] == 1024:
            pytest.skip("Skipping this test temporarily.")
        self.run_generate_test(**params)


instantiate_parametrized_tests(TestFalconOptimize)

if __name__ == "__main__":
    run_tests()
