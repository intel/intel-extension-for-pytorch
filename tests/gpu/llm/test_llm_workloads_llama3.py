from test_llm_common_utils import TestIpexLLMOptimizeBase
from test_llm_common_utils import (
    curpath,
    common_params,
    is_data_center_gpu,
)
import transformers
import pytest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    parametrize,
)


class TestLlama3Optimize(TestIpexLLMOptimizeBase):
    model_class = transformers.models.llama.modeling_llama.LlamaForCausalLM
    model_config = {"config_path": f"{curpath}/hf_configs/llama3"}

    @parametrize("use_static_cache", common_params["use_static_cache"])
    @parametrize("num_beams", common_params["num_beams"])
    @parametrize("input_tokens_length", common_params["input_tokens_length"])
    @parametrize("max_new_tokens", common_params["max_new_tokens"])
    def test_llama3_model_generate(self, **params):
        if not is_data_center_gpu():
            pytest.skip("Skipping this test on platforms except data centor GPU.")
        self.run_generate_test(**params)


instantiate_parametrized_tests(TestLlama3Optimize)

if __name__ == "__main__":
    run_tests()
