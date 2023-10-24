import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase

batch_size = 5
beam_size = 4
vocab_size = 50401

pad_token_id = 0
eos_token_id = -1
time_step = 3
length_penalty = 1.0
input_length = 32
max_out_length = 32
do_early_stopping = True

finish_tensor = torch.zeros([batch_size]).bool().to("xpu")
sequence_lengths = torch.zeros(
    (batch_size * beam_size), dtype=torch.long, device=torch.device("xpu")
)
beam_hyps_num_beams = torch.zeros(
    (batch_size), dtype=torch.long, device=torch.device("xpu")
)
beam_hyps_output_ids_tgt = torch.empty(
    (2 * batch_size * beam_size, max_out_length),
    dtype=torch.long,
    device=torch.device("xpu"),
)
word_ids = torch.empty(
    (max_out_length, batch_size * beam_size),
    dtype=torch.long,
    device=torch.device("xpu"),
)
beam_ids = torch.empty(
    (max_out_length, batch_size * beam_size),
    dtype=torch.long,
    device=torch.device("xpu"),
)
beam_hyps_sequence_lengths_tgt = torch.zeros(
    (2 * batch_size * beam_size), dtype=torch.long, device=torch.device("xpu")
)
output_token_ids = torch.empty(
    (max_out_length, batch_size * beam_size),
    dtype=torch.long,
    device=torch.device("xpu"),
)
output_beam_ids = torch.empty(
    (max_out_length, batch_size * beam_size),
    dtype=torch.long,
    device=torch.device("xpu"),
)


class testBeamsearch(TestCase):
    def test_beam_search_topk_float(self):
        input_score = torch.randn(batch_size * beam_size, vocab_size).to("xpu")
        next_token_scores, next_tokens = torch.topk(
            input_score.view(batch_size, beam_size * vocab_size),
            beam_size,
            dim=1,
            largest=True,
            sorted=True,
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor").view(
            -1
        )
        next_tokens = (next_tokens % vocab_size).view(-1)
        # print(next_token_scores.cpu())
        # print(next_tokens.cpu())
        # print(next_indices.cpu())

        beam_hyps_normed_scores = torch.empty(
            (2 * batch_size * beam_size),
            dtype=input_score.dtype,
            device=torch.device("xpu"),
        )
        beam_hyps_min_normed_scores = torch.empty(
            (batch_size), dtype=input_score.dtype, device=torch.device("xpu")
        )
        beam_hyps_score = torch.empty(
            (2 * batch_size * beam_size),
            dtype=input_score.dtype,
            device=torch.device("xpu"),
        )
        score, token_idx, beam_idx = torch.ops.torch_ipex.beam_search_topk(
            input_score,
            finish_tensor,
            pad_token_id,
            eos_token_id,
            length_penalty,
            beam_size,
            batch_size,
            vocab_size,
            time_step,
            do_early_stopping,
            input_length,
            max_out_length,
            output_token_ids,
            output_beam_ids,
            beam_hyps_num_beams,
            beam_hyps_normed_scores,
            beam_hyps_min_normed_scores,
            beam_hyps_output_ids_tgt,
            beam_hyps_sequence_lengths_tgt,
            beam_hyps_score,
        )
        # print(score.cpu())
        # print(token_idx.cpu())
        # print(beam_idx.cpu())
        self.assertEqual(next_indices.cpu(), beam_idx.cpu())
        self.assertEqual(next_tokens.cpu(), token_idx.cpu())

    def test_beam_search_topk_half(self):
        input_score = (
            torch.randn(batch_size * beam_size, vocab_size).to(torch.half).to("xpu")
        )
        next_token_scores, next_tokens = torch.topk(
            input_score.view(batch_size, beam_size * vocab_size),
            beam_size,
            dim=1,
            largest=True,
            sorted=True,
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor").view(
            -1
        )
        next_tokens = (next_tokens % vocab_size).view(-1)
        # print(next_token_scores.cpu())
        # print(next_tokens.cpu())
        # print(next_indices.cpu())

        beam_hyps_normed_scores = torch.empty(
            (2 * batch_size * beam_size),
            dtype=input_score.dtype,
            device=torch.device("xpu"),
        )
        beam_hyps_min_normed_scores = torch.empty(
            (batch_size), dtype=input_score.dtype, device=torch.device("xpu")
        )
        beam_hyps_score = torch.empty(
            (2 * batch_size * beam_size),
            dtype=input_score.dtype,
            device=torch.device("xpu"),
        )
        score, token_idx, beam_idx = torch.ops.torch_ipex.beam_search_topk(
            input_score,
            finish_tensor,
            pad_token_id,
            eos_token_id,
            length_penalty,
            beam_size,
            batch_size,
            vocab_size,
            time_step,
            do_early_stopping,
            input_length,
            max_out_length,
            output_token_ids,
            output_beam_ids,
            beam_hyps_num_beams,
            beam_hyps_normed_scores,
            beam_hyps_min_normed_scores,
            beam_hyps_output_ids_tgt,
            beam_hyps_sequence_lengths_tgt,
            beam_hyps_score,
        )
        # print(score.cpu())
        # print(token_idx.cpu())
        # print(beam_idx.cpu())
        self.assertEqual(next_indices.cpu(), beam_idx.cpu())
        self.assertEqual(next_tokens.cpu(), token_idx.cpu())

    def test_beam_search_topk_bfl6(self):
        input_score = (
            torch.randn(batch_size * beam_size, vocab_size).to(torch.bfloat16).to("xpu")
        )
        next_token_scores, next_tokens = torch.topk(
            input_score.view(batch_size, beam_size * vocab_size),
            beam_size,
            dim=1,
            largest=True,
            sorted=True,
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor").view(
            -1
        )
        next_tokens = (next_tokens % vocab_size).view(-1)
        # print(next_token_scores.cpu())
        # print(next_tokens.cpu())
        # print(next_indices.cpu())

        beam_hyps_normed_scores = torch.empty(
            (2 * batch_size * beam_size),
            dtype=input_score.dtype,
            device=torch.device("xpu"),
        )
        beam_hyps_min_normed_scores = torch.empty(
            (batch_size), dtype=input_score.dtype, device=torch.device("xpu")
        )
        beam_hyps_score = torch.empty(
            (2 * batch_size * beam_size),
            dtype=input_score.dtype,
            device=torch.device("xpu"),
        )
        score, token_idx, beam_idx = torch.ops.torch_ipex.beam_search_topk(
            input_score,
            finish_tensor,
            pad_token_id,
            eos_token_id,
            length_penalty,
            beam_size,
            batch_size,
            vocab_size,
            time_step,
            do_early_stopping,
            input_length,
            max_out_length,
            output_token_ids,
            output_beam_ids,
            beam_hyps_num_beams,
            beam_hyps_normed_scores,
            beam_hyps_min_normed_scores,
            beam_hyps_output_ids_tgt,
            beam_hyps_sequence_lengths_tgt,
            beam_hyps_score,
        )
        # print(score.cpu())
        # print(token_idx.cpu())
        # print(beam_idx.cpu())
        self.assertEqual(next_indices.cpu(), beam_idx.cpu())
        self.assertEqual(next_tokens.cpu(), token_idx.cpu())

    def test_beam_search_update_cache_idx(self):
        input_cache = torch.tensor(
            [[2, 1, 1, 0], [1, 1, 0, 0], [3, 1, 1, 2], [2, 0, 1, 0]], dtype=torch.int64
        ).to("xpu")
        beam_idx = torch.tensor([1, 2, 3, 0], dtype=torch.int64).to("xpu")
        out = torch.ops.torch_ipex.update_beam_indices_for_cache(
            input_cache, beam_idx, 4, 1
        )
        ref_output = torch.tensor(
            [[1, 1, 0, 2], [1, 0, 0, 1], [1, 1, 2, 3], [0, 1, 0, 2], [0, 1, 2, 3]],
            dtype=torch.int64,
        )
        print(out.cpu())
        self.assertEqual(out.cpu(), ref_output)
