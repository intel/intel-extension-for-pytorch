import unittest, copy
from itertools import product

import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase

class TestRNNTUpdateBatch(TestCase):
    def _test_org(self, hidden, hidden_prime, x, batch_size, max_symbol, blank_id, loop_cnt):
        f = x[:, 0, :].unsqueeze(1)

        max_lens = torch.tensor([self.max_len-1 for i in range(batch_size)], dtype=torch.int64)

        time_idxs = torch.zeros((batch_size), dtype=torch.long)
        out_lens = torch.tensor([i for i in range(batch_size)], dtype=torch.long)
        blank_vec = torch.zeros((batch_size), dtype=torch.long)
        not_blank = torch.zeros((batch_size), dtype=torch.int)
        symbols_added = torch.zeros((batch_size), dtype=torch.long)
        label_row = torch.tensor([i for i in range(batch_size)])
        label_col = torch.zeros((batch_size), dtype=torch.long)
        label_tensor = torch.tensor([self._SOS]).repeat(batch_size, self.max_len*max_symbol)

        k = torch.tensor([i for i in range(batch_size)], dtype=torch.long)
        # TODO: randomly set k to blank_id (28)

        for i in range(loop_cnt):
            blankness = k.eq(blank_id)
            time_idxs = time_idxs + blankness
            # it doesn't matter if blank_vec is update now or later,
            # tmp_blank_vec always get correct value for this round
            blank_vec = time_idxs.ge(out_lens)

            if blank_vec.nonzero().size(0) == batch_size:
                break

            symbols_added *= blankness.logical_not()
            tmp_blank_vec = blank_vec.logical_or(blankness)
            not_blank = tmp_blank_vec.eq(0)
            idx = (not_blank).nonzero(as_tuple=True)[0]

            hidden[0][:, idx, :] = hidden_prime[0][:, idx, :]
            hidden[1][:, idx, :] = hidden_prime[1][:, idx, :]

            label_col += not_blank
            label_tensor.index_put_([label_row, label_col], (k-self._SOS)*not_blank, accumulate=True)

            symbols_added += not_blank

            need_add = symbols_added.ge(max_symbol)

            time_idxs += need_add

            blankness.logical_or_(need_add)
            temp = symbols_added.lt(max_symbol)
            symbols_added *= temp

            if blankness.nonzero().size(0) > 0:
                fetch_time_idxs = time_idxs.min(max_lens)
                f = x[list(range(x.size(0))), fetch_time_idxs, :].unsqueeze(1)

        return blank_vec, blankness, label_col, time_idxs, symbols_added, not_blank, label_tensor, hidden, f

    def _test_rnnt_update_batch_kernel(self, hidden, hidden_prime, x, batch_size, max_symbol, blank_id, loop_cnt):
        f = x[:, 0, :].unsqueeze(1)

        time_idxs = torch.zeros((batch_size), dtype=torch.int)
        out_lens = torch.tensor([i for i in range(batch_size)], dtype=torch.int)
        blank_vec_out = torch.zeros((batch_size), dtype=torch.int)
        not_blank = torch.zeros((batch_size), dtype=torch.int)
        blankness = torch.zeros((batch_size), dtype=torch.int)

        symbols_added = torch.zeros((batch_size), dtype=torch.int)
        label_col = torch.zeros((batch_size), dtype=torch.int)
        label_tensor = torch.empty((batch_size, self.max_len*max_symbol), dtype=torch.long).fill_(self._SOS)

        k = torch.tensor([i for i in range(batch_size)], dtype=torch.long)
        label_to_put = torch.zeros((batch_size), dtype=torch.long)

        label_for_next_loop = torch.tensor([self._SOS for i in range(batch_size)], dtype=torch.long)
        for i in range(loop_cnt):
            finished = torch.ops.torch_ipex.rnnt_update_batch(
                k,
                out_lens,
                label_col,
                symbols_added,
                time_idxs,
                blankness,
                blank_vec_out,
                not_blank,
                label_to_put,
                label_tensor,
                label_for_next_loop,
                hidden[0],
                hidden[1],
                hidden_prime[0],
                hidden_prime[1],
                x,
                f,
                max_symbol,
                blank_id,
                batch_size,
                self._SOS,
                self.max_len)

            if finished:
                break

        return blank_vec_out, blankness, label_col, time_idxs, symbols_added, not_blank, label_tensor, hidden, f

    def test_rnnt_update_batch(self):
        self._SOS = -1
        self.max_len = 192
        dtypes = [torch.float, torch.bfloat16]
        loop_cnts = [1, 10, 30]
        batch_sizes = [1, 15, 64, 448]
        max_symbols = [30]
        blank_ids = [1, 21]

        for batch_size, max_symbol, blank_id, loop_cnt, dtype in list(product(batch_sizes, max_symbols, blank_ids, loop_cnts, dtypes)):

            x_org = torch.randn([self.max_len, batch_size, 2], dtype=dtype)
            x = copy.deepcopy(x_org)
            hidden = [torch.zeros([2, batch_size, 320], dtype=dtype), torch.zeros([2, batch_size, 320], dtype=dtype)]
            hidden_prime = [torch.randn([2, batch_size, 320], dtype=dtype), torch.randn([2, batch_size, 320], dtype=dtype)]

            blank_vec_org, blankness_org, label_col_org, time_idxs_org, symbols_added_org, not_blank_org, label_tensor_org, hidden_org, f_org = self._test_org(hidden, hidden_prime, x_org.transpose(0, 1), batch_size, max_symbol, blank_id, loop_cnt)
            blank_vec_out, blankness_out, label_col, time_idxs, symbols_added, not_blank, label_tensor, hidden, f = self._test_rnnt_update_batch_kernel(hidden, hidden_prime, x.transpose(0,1), batch_size, max_symbol, blank_id, loop_cnt)
            self.assertEqual(blank_vec_org, blank_vec_out)
            self.assertEqual(blankness_org, blankness_out)
            self.assertEqual(label_col_org, label_col)
            self.assertEqual(time_idxs_org, time_idxs)
            self.assertEqual(symbols_added_org, symbols_added)
            self.assertEqual(not_blank_org, not_blank)
            self.assertEqual(label_tensor_org, label_tensor)
            self.assertEqual(hidden_org, hidden)
            self.assertEqual(f_org, f)

class TestRNNTEmbedding(TestCase):
    def _test_org(self, y, embedding):
        y_mask = y.eq(self._SOS)
        y.masked_fill_(y_mask, 0)
        y = embedding(y)
        y.masked_fill_(y_mask.unsqueeze(2), 0.0)
        return y

    def _test_rnnt_embedding_kernel(self, y_in, embedding):
        batch_size = y_in.shape[0]
        embedding_dim = embedding.weight.shape[1]
        y = torch.zeros([batch_size, y_in.shape[1], embedding_dim], dtype=embedding.weight.dtype)

        torch.ops.torch_ipex.rnnt_embedding(
            embedding.weight,
            y_in,
            y,
            self._SOS,
            batch_size,
            embedding_dim)
        return y

    def test_rnnt_embedding(self):
        self._SOS = -1
        for dtype in [torch.float, torch.bfloat16]:
            vocab_size = 29
            pred_n_hidden = 320
            embedding = torch.nn.Embedding(vocab_size - 1, pred_n_hidden).to(dtype)
            y_org = torch.Tensor([-1, 2, 15, -1, 5]).unsqueeze(1).to(torch.long)

            y = copy.deepcopy(y_org)

            y_embed_org = self._test_org(y_org, embedding)
            y_embed = self._test_rnnt_embedding_kernel(y, embedding)

            self.assertEqual(y_embed_org, y_embed)

if __name__ == '__main__':
    test = unittest.main()
