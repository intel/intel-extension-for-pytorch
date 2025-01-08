from test_llm_common_utils import TestIpexLLMOptimizeBase
from test_llm_common_utils import (
    curpath,
    common_params,
)
import transformers
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    parametrize,
)


class TestBloomOptimize(TestIpexLLMOptimizeBase):
    model_class = transformers.models.bloom.modeling_bloom.BloomForCausalLM
    model_config = {"config_path": f"{curpath}/hf_configs/bloom"}

    @parametrize("use_static_cache", common_params["use_static_cache"])
    @parametrize("num_beams", common_params["num_beams"])
    @parametrize("input_tokens_length", common_params["input_tokens_length"])
    @parametrize("max_new_tokens", common_params["max_new_tokens"])
    def test_bloom_model_generate(self, **params):
        self.run_generate_test(**params)


instantiate_parametrized_tests(TestBloomOptimize)

if __name__ == "__main__":
    run_tests()
