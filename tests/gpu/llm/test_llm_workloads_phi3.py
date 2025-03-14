from test_llm_common_utils import TestIpexLLMOptimizeBase
from test_llm_common_utils import (
    curpath,
    common_params,
    is_data_center_gpu,
)
import transformers
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    parametrize,
)


class TestPhi3Optimize(TestIpexLLMOptimizeBase):
    model_class = transformers.models.phi3.modeling_phi3.Phi3ForCausalLM
    model_config = {"config_path": f"{curpath}/hf_configs/phi3"}
    is_data_center_gpu = is_data_center_gpu()
    input_tokens_length = common_params["input_tokens_length"] if is_data_center_gpu else common_params["input_tokens_length_for_win"]
    max_new_tokens = common_params["max_new_tokens"] if is_data_center_gpu else common_params["max_new_tokens_for_win"]

    @parametrize("use_static_cache", common_params["use_static_cache"])
    @parametrize("num_beams", common_params["num_beams"])
    @parametrize("input_tokens_length", input_tokens_length)
    @parametrize("max_new_tokens", max_new_tokens)
    def test_phi3_model_generate(self, **params):
        self.run_generate_test(**params)


instantiate_parametrized_tests(TestPhi3Optimize)

if __name__ == "__main__":
    run_tests()

