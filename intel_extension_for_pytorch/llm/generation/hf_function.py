from intel_extension_for_pytorch.transformers.generation.beam_search import _beam_search
from intel_extension_for_pytorch.transformers.generation.greedy_search import (
    _greedy_search,
)
from intel_extension_for_pytorch.transformers.generation.sample import (
    _sample,
)

hf_greedy_search = _greedy_search
hf_beam_search = _beam_search
hf_sample = _sample
