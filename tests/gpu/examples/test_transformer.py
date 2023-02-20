import torch
import intel_extension_for_pytorch # noqa

from torch.testing._internal.common_utils import TestCase
import contextlib
from torch import nn
import numpy as np
from torch.testing._internal.common_utils import TEST_WITH_CROSSREF


class TestTorchMethod(TestCase):

    def test_transformerencoderlayer(self, device="xpu", dtype=torch.float):
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        atol = 1e-05
        rtol = 1e-07
        if 'xpu' in device:
            atol = 0.001
            rtol = 0.01

        def _test(training, batch_first, atol, rtol):

            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x
            model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first, device=device, dtype=dtype)
            if not training:
                assert dropout == 0
                model = model.eval()
            for (idx, p) in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)
            encoder_input = torch.tensor([[[20.0, 30.0, 40.0, 50.0]]], device=device, dtype=dtype)
            result = model(encoder_input)
            ref_output = torch.tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]], device=device, dtype=dtype)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            mask = torch.tensor([[0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            mask = torch.tensor([[1]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            result = result.cpu().detach().numpy()
            self.assertTrue(np.isnan(result).all())

            encoder_input = perm_fn(torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.272644, 0.119035, -0.691669, 0.153486]], [[2.272644, 0.119035, -0.691669, 0.153486]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            mask = torch.tensor([[0, 0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            mask = torch.tensor([[1, 0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.301516, 0.092249, -0.679101, 0.103088]], [[2.301516, 0.092249, -0.679101, 0.103088]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891], [0.5387, 0.1655, 0.3565, 0.0471]], [[0.8335, 0.2799, 0.5031, 0.2947], [0.1402, 0.0318, 0.7636, 0.1346]], [[0.6333, 0.9344, 0.1376, 0.9938], [0.8924, 0.2872, 0.6692, 0.2944]], [[0.9897, 0.6915, 0.3154, 0.1733], [0.8645, 0.3513, 0.3064, 0.0767]], [[0.8117, 0.2366, 0.4838, 0.7881], [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249], [2.427987, 0.021213, -0.602496, -0.084103]], [[2.424689, 0.019155, -0.604793, -0.085672], [2.413863, 0.022211, -0.612486, -0.07249]], [[2.433774, 0.021598, -0.598343, -0.087548], [2.425104, 0.019748, -0.604515, -0.084839]], [[2.436185, 0.022682, -0.596625, -0.087261], [2.433556, 0.021891, -0.598509, -0.086832]], [[2.416246, 0.017512, -0.610712, -0.082961], [2.422901, 0.024187, -0.606178, -0.074929]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            mask = torch.zeros([2, 5], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642], [2.428811, 0.021445, -0.601912, -0.084252]], [[2.425009, 0.019155, -0.604566, -0.085899], [2.415408, 0.02249, -0.611415, -0.073]], [[2.434199, 0.021682, -0.598039, -0.087699], [2.42598, 0.019941, -0.603896, -0.085091]], [[2.436457, 0.022736, -0.59643, -0.08736], [2.434021, 0.022093, -0.598179, -0.08679]], [[2.416531, 0.017498, -0.610513, -0.083181], [2.4242, 0.024653, -0.605266, -0.074959]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result.cpu(), ref_output.cpu(), atol=atol, rtol=rtol)

        for batch_first in (True, False):
            for training in (True, False):
                if training:
                    cm = contextlib.nullcontext()
                else:
                    cm = torch.no_grad()
                with cm:
                    _test(batch_first=batch_first, training=training, atol=atol, rtol=rtol)

    def test_transformerdecoderlayer(self):
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        seq_length = 5
        tgt_length = 3
        for batch_first in (False, True):

            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x
            model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
            for (idx, p) in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)
            decoder_input = torch.tensor([[[20.0, 30.0, 40.0, 50.0]]])
            memory_input = torch.tensor([[[60.0, 70.0, 80.0, 90.0]]])
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor([[[2.314351, 0.094805, -0.671322, 0.101977]]])
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-05)

            decoder_input = perm_fn(torch.tensor([[[9.0, 10.0, 11.0, 12.0]], [[11.0, 12.0, 13.0, 14.0]]]))
            memory_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
            result = model(decoder_input, memory_input)
            result = result.detach().numpy()
            ref_output = perm_fn(torch.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]], [[2.422245, 0.051716, -0.606338, -0.024756]]]))
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-05)

            decoder_input = perm_fn(torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]))
            memory_input = perm_fn(torch.tensor([[[9.0, 10.0, 11.0, 12.0]], [[11.0, 12.0, 13.0, 14.0]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]], [[2.343536, 0.085561, -0.654954, 0.074991]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-05)

            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034], [0.2678, 0.3677, 0.4459, 0.7166]], [[0.81, 0.3716, 0.4096, 0.1976], [0.6958, 0.8844, 0.6081, 0.8315]], [[0.0494, 0.9343, 0.5955, 0.383], [0.5404, 0.3464, 0.9378, 0.62]]]))
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891], [0.5387, 0.1655, 0.3565, 0.0471]], [[0.8335, 0.2799, 0.5031, 0.2947], [0.1402, 0.0318, 0.7636, 0.1346]], [[0.6333, 0.9344, 0.1376, 0.9938], [0.8924, 0.2872, 0.6692, 0.2944]], [[0.9897, 0.6915, 0.3154, 0.1733], [0.8645, 0.3513, 0.3064, 0.0767]], [[0.8117, 0.2366, 0.4838, 0.7881], [0.3718, 0.4945, 0.9511, 0.0864]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096], [2.431935, 0.028907, -0.599809, -0.072488]], [[2.428457, 0.027053, -0.602275, -0.073462], [2.43197, 0.029387, -0.599789, -0.071621]], [[2.431934, 0.028196, -0.599802, -0.073809], [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-05)

            key_padding_mask = torch.zeros(2, 3) == 1
            result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096], [2.431935, 0.028907, -0.599809, -0.072488]], [[2.428457, 0.027053, -0.602275, -0.073462], [2.43197, 0.029387, -0.599789, -0.071621]], [[2.431934, 0.028196, -0.599802, -0.073809], [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-05)
