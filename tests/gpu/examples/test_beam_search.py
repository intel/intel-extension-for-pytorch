import torch
import intel_extension_for_pytorch

input_cache = torch.tensor([[2, 1, 1, 0], [1, 1, 0, 0], [3, 1, 1, 2], [2, 0, 1, 0]], dtype=torch.int64).to("xpu")
beam_idx = torch.tensor([1, 2, 3, 0], dtype=torch.int64).to("xpu")
out = torch.ops.torch_ipex.update_beam_indices_for_cache(input_cache, beam_idx, 4, 1)
print(out.cpu())


input_score = torch.load("/4T-720/majing/ipex2.0/score.pt").to("xpu")
print(input_score.size())


vocab_size = input_score.size(1)
next_token_scores = input_score.view(1, 4 * vocab_size)
next_token_scores, next_tokens = torch.topk(
    next_token_scores, 2 * 4, dim=1, largest=True, sorted=True
)
next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
next_tokens = next_tokens % vocab_size
print(next_indices.cpu())
print(next_tokens.cpu())
pad_token_id = 0
eos_token_id = vocab_size + 1
batch_size = 1
beam_size = 4
time_step = 3
length_penalty = 1.0
input_length = 32
max_out_length = 32

finish_tensor = torch.tensor([False, False, False, False]).to("xpu")
sequence_lengths = torch.zeros((batch_size * beam_size), dtype=torch.long, device=input_score.device)
beam_hyps_num_beams = torch.zeros((batch_size), dtype=torch.long, device=input_score.device)
beam_hyps_normed_scores = torch.empty((2 * batch_size * beam_size), dtype=input_score.dtype, device=input_score.device)
beam_hyps_min_normed_scores = torch.empty((batch_size), dtype=input_score.dtype, device=input_score.device)
beam_hyps_output_ids_tgt = torch.empty((2 * batch_size * beam_size, max_out_length), dtype=torch.long, device=input_score.device)
word_ids = torch.empty((max_out_length, batch_size * beam_size), dtype=torch.long, device=input_score.device)
beam_ids = torch.empty((max_out_length, batch_size * beam_size), dtype=torch.long, device=input_score.device)
beam_hyps_sequence_lengths_tgt = torch.zeros((2 * batch_size * beam_size), dtype=torch.long, device=input_score.device)
beam_hyps_score = torch.empty((2 * batch_size * beam_size), dtype=input_score.dtype, device=input_score.device)

score, index = torch.ops.torch_ipex.beam_search_topk(input_score, finish_tensor,
pad_token_id, eos_token_id, length_penalty, beam_size, batch_size, vocab_size,
time_step, input_length, max_out_length, beam_hyps_num_beams, beam_hyps_normed_scores,
beam_hyps_min_normed_scores, beam_hyps_output_ids_tgt, word_ids, beam_ids,
beam_hyps_sequence_lengths_tgt, sequence_lengths, beam_hyps_score)
print(score.cpu())
print(index.cpu())


torch.ops.torch_ipex.update_output_indices(index, beam_ids, word_ids, finish_tensor, sequence_lengths,
beam_hyps_num_beams, time_step, batch_size, beam_size, vocab_size, eos_token_id)
print(beam_ids[time_step, :].cpu())
print(word_ids[time_step, :].cpu())