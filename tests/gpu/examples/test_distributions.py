"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in # 4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).
"""

import math
from collections import namedtuple

import torch
from torch.autograd import gradcheck, Variable
from torch.distributions import (
    Bernoulli,
    Exponential,
    Multinomial,
    Normal,
    Uniform,
    Dirichlet,
    Poisson,
)
from torch.testing._internal.common_utils import TestCase, load_tests, set_rng_seed

import intel_extension_for_pytorch  # noqa

import pytest
import scipy

#  backup default dtype
dtype_origin = torch.get_default_dtype()

#  load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
#  sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_NUMPY = True
try:
    import scipy.special
    import scipy.stats
except ImportError:
    TEST_NUMPY = False


def pairwise(Dist, *params):
    """
    Creates a pair of distributions `Dist` initialized to test each element of
    param with each other.
    """
    params1 = [torch.tensor([p] * len(p), dtype=torch.double) for p in params]
    params2 = [p.transpose(0, 1) for p in params1]
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    """
    Checks if all entries of a tensor is nan.
    """
    return (tensor != tensor).all()


#  Register all distributions for generic tests.
Example = namedtuple("Example", ["Dist", "params"])
EXAMPLES = [
    Example(
        Bernoulli,
        [
            {
                "probs": torch.tensor(
                    [0.7, 0.2, 0.4], dtype=torch.double, requires_grad=True
                )
            },
            {"probs": torch.tensor([0.3], dtype=torch.double, requires_grad=True)},
            {"probs": 0.3},
            {"logits": torch.tensor([0.0], dtype=torch.double, requires_grad=True)},
        ],
    ),
    Example(
        Dirichlet,
        [
            {"concentration": torch.randn(2, 3).exp().requires_grad_()},
            {"concentration": torch.randn(4).exp().requires_grad_()},
        ],
    ),
]

BAD_EXAMPLES = [
    Example(
        Bernoulli,
        [
            {
                "probs": torch.tensor(
                    [1.1, 0.2, 0.4], dtype=torch.double, requires_grad=True
                )
            },
            {"probs": torch.tensor([-0.5], dtype=torch.double, requires_grad=True)},
            {"probs": 1.00001},
        ],
    )
]

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestDistributions(TestCase):
    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        torch.set_default_dtype(torch.double)
        #  performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()
        if s.is_floating_point():
            s = s.detach().requires_grad_()

        expected_shape = distribution.batch_shape + distribution.event_shape
        self.assertEqual(s.size(), expected_shape)

        def apply_fn(s, *params):
            return dist_ctor(*params, validate_args=False).log_prob(s)

        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)
        torch.set_default_dtype(dtype_origin)

    def _check_log_prob(self, dist, asset_fn):
        torch.set_default_dtype(torch.double)
        #  checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        log_probs_data_flat = log_probs.view(-1)
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            asset_fn(i, val.squeeze(), log_prob)
        torch.set_default_dtype(dtype_origin)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_bernoulli(self):
        torch.set_default_dtype(torch.double)
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        p_dpcpp = torch.tensor([0.7, 0.2, 0.4], requires_grad=True, device=sycl_device)

        r = torch.tensor(0.3, requires_grad=True)
        r_dpcpp = torch.tensor(0.3, requires_grad=True, device=sycl_device)
        s = 0.3
        self.assertEqual(Bernoulli(p_dpcpp).sample((8,)).size(), (8, 3))
        self.assertFalse(Bernoulli(p_dpcpp).sample().requires_grad)
        self.assertEqual(Bernoulli(r_dpcpp).sample((8,)).size(), (8,))
        self.assertEqual(Bernoulli(r_dpcpp).sample().size(), ())
        self.assertEqual(
            Bernoulli(r_dpcpp).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self.assertEqual(Bernoulli(s).sample().size(), ())
        self._gradcheck_log_prob(Bernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p_dpcpp), ref_log_prob)
        self._check_log_prob(
            Bernoulli(logits=p_dpcpp.log() - (-p_dpcpp).log1p()), ref_log_prob
        )
        self.assertRaises(NotImplementedError, Bernoulli(r_dpcpp).rsample)

        #  check entropy computation
        #  TO DO: implement dpcpp entropy
        # self.assertEqual(Bernoulli(p_dpcpp).entropy(), torch.tensor([0.6108, 0.5004, 0.6730]), prec=1e-4)
        self.assertEqual(Bernoulli(torch.tensor([0.0])).entropy(), torch.tensor([0.0]))
        self.assertEqual(
            Bernoulli(s).entropy(), torch.tensor(0.6108), atol=1e-4, rtol=0
        )
        torch.set_default_dtype(dtype_origin)

    def test_log_normal(self):
        torch.set_default_dtype(torch.double)
        for device in torch.testing._internal.common_device_type.get_all_device_types():
            a = torch.tensor([10], dtype=torch.float, device=device).log_normal_()
            self.assertEqual(a.dtype, torch.float)
            self.assertEqual(a.size(), torch.Size([1]))
        torch.set_default_dtype(dtype_origin)

    def test_exponential(self):
        torch.set_default_dtype(torch.double)
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

        self._gradcheck_log_prob(Exponential, (rate,))
        state = torch.get_rng_state()
        eps = rate.new(rate.size()).exponential_()
        torch.set_rng_state(state)
        z = Exponential(rate).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(rate.grad, -eps / rate**2)
        rate.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = rate.view(-1)[idx]
            expected = math.log(m) - m * x
            self.assertAlmostEqual(log_prob, expected, places=3)

        self._check_log_prob(Exponential(rate), ref_log_prob)
        torch.set_default_dtype(dtype_origin)

    def test_multinomial_1d(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True, device=sycl_device)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (3,))
        self.assertEqual(Multinomial(total_count, p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(Multinomial(total_count, p).sample((1,)).size(), (1, 3))
        # self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        # self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, Multinomial(10, p).rsample)

    def test_multinomial_1d_log_prob(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True, device=sycl_device)
        dist = Multinomial(total_count, probs=p)
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.cpu().numpy(), n=total_count, p=dist.probs.detach().cpu().numpy()
            ),
            dtype=x.dtype,
        )
        self.assertEqual(log_prob, expected)

        dist = Multinomial(total_count, logits=p.log())
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.cpu().numpy(), n=total_count, p=dist.probs.detach().cpu().numpy()
            ),
            dtype=x.dtype,
        )
        self.assertEqual(log_prob, expected)

    def test_multinomial_2d(self):
        total_count = 10
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True, device=sycl_device)
        s = torch.tensor(probabilities_1, requires_grad=True, device=sycl_device)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (2, 3))
        self.assertEqual(
            Multinomial(total_count, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        self.assertEqual(Multinomial(total_count, p).sample((6,)).size(), (6, 2, 3))
        set_rng_seed(0)
        # self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        # self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])

        #  sample check for extreme value of probs
        # self.assertEqual(Multinomial(total_count, s).sample().to(cpu_device), torch.tensor([[total_count, 0], [0, total_count]]))

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_normal(self):
        loc = torch.randn(5, 5, requires_grad=True, device=sycl_device)
        scale = torch.randn(5, 5).abs().requires_grad_().to("xpu")
        loc_1d = torch.randn(1, requires_grad=True, device=sycl_device)
        scale_1d = torch.randn(1).abs().requires_grad_().to("xpu")
        loc_delta = torch.tensor([1.0, 0.0], device=sycl_device)
        scale_delta = torch.tensor([1e-5, 1e-5], device=sycl_device)
        self.assertEqual(Normal(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Normal(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Normal(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Normal(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Normal(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(Normal(-0.7, 50.0).sample((1,)).size(), (1,))

        #  sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            Normal(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            rtol=1e-4,
            atol=1e-4,
        )

        # self._gradcheck_log_prob(Normal, (loc, scale))
        # self._gradcheck_log_prob(Normal, (loc, 1.0))
        # self._gradcheck_log_prob(Normal, (0.0, scale))

        state = torch.get_rng_state()
        eps = torch.normal(torch.zeros_like(loc), torch.ones_like(scale))
        torch.set_rng_state(state)
        z = Normal(loc, scale).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        # self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        #  scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = math.exp(-((x - m) ** 2) / (2 * s**2)) / math.sqrt(
                2 * math.pi * s**2
            )
            self.assertAlmostEqual(log_prob, math.log(expected), places=3)

        # self._check_log_prob(Normal(loc, scale), ref_log_prob)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True).to(sycl_device)
        high = (torch.ones(5, 5) * 3).requires_grad_().to(sycl_device)
        low_1d = torch.zeros(1, requires_grad=True).to(sycl_device)
        high_1d = (torch.ones(1) * 3).requires_grad_().to(sycl_device)
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        #  Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        # self.assertEqual(uniform.log_prob(above_high).item(), -inf, allow_inf=True)
        # self.assertEqual(uniform.log_prob(below_low).item(), -inf, allow_inf=True)

        #  check cdf computation when value outside range
        # self.assertEqual(uniform.cdf(below_low).item(), 0)
        # self.assertEqual(uniform.cdf(above_high).item(), 1)

        set_rng_seed(1)
        # self._gradcheck_log_prob(Uniform, (low, high))
        # self._gradcheck_log_prob(Uniform, (low, 1.0))
        # self._gradcheck_log_prob(Uniform, (0.0, high))

        state = torch.get_rng_state()
        rand = torch.empty(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        # self.assertEqual(low.grad, 1 - rand)
        # self.assertEqual(high.grad, rand)
        #  low.grad.zero_()
        #  high.grad.zero_()

    def test_s_standard_gamma(self):
        alpha = torch.randn(50, 60).exp()
        alpha_1d = torch.randn(1).exp()
        self.assertEqual(torch._standard_gamma(alpha).size(), (50, 60))
        self.assertEqual(torch._standard_gamma(alpha_1d).size(), (1,))

        beta = torch.ones_like(alpha)

        def ref_log_prob(idx, x, log_prob):
            a = alpha.view(-1)[idx].detach()
            b = beta.view(-1)[idx].detach()
            expected = scipy.stats.gamma.logpdf(x.to("cpu"), a, scale=1 / b)
            self.assertEqual(log_prob.to("cpu"), expected, atol=1e-3, rtol=0)

        self._check_log_prob(
            torch.distributions.Gamma(alpha.to("xpu"), beta.to("xpu")), ref_log_prob
        )

    def test_s_standard_gamma_grad(self):
        alpha_cpu = torch.randn(2, 3, requires_grad=True).abs()
        alpha_xpu = alpha_cpu.to("xpu")
        s_cpu = torch._standard_gamma(alpha_cpu)
        s_xpu = torch._standard_gamma(alpha_xpu)
        print("s_cpu: ", s_cpu)
        print("s_xpu: ", s_xpu.to("cpu"))

        grad_cpu = Variable(torch.sum(s_cpu), requires_grad=True)
        grad_cpu.backward()
        print("grad_cpu: ", grad_cpu.grad)

        grad_xpu = Variable(torch.sum(s_xpu), requires_grad=True)
        grad_xpu.backward()
        print("grad_xpu: ", grad_xpu.grad.to("cpu"))
        self.assertEqual(grad_xpu.grad.to("cpu"), grad_cpu.grad)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_binomial_log_prob(self):
        for prop in [0.0, 0.5, 0.3, 0.05, 0.02, 0.75, 0.9, 1.0]:
            total_count = torch.tensor([[8, 70, 100], [3000, 80000, 700000]]).to("xpu")
            bin0 = torch.distributions.Binomial(
                total_count, torch.tensor(1.0).to("xpu")
            )
            self.assertEqualIgnoreType(bin0.sample().to("cpu"), total_count.to("cpu"))
            bin1 = torch.distributions.Binomial(
                total_count, torch.tensor(prop).to("xpu")
            )
            samples = bin1.sample(torch.Size((100000,)))
            self.assertTrue((samples <= total_count.type_as(samples)).all())
            self.assertEqual(
                samples.mean(dim=0) / total_count,
                bin1.mean / total_count,
                atol=0.02,
                rtol=0,
            )
            self.assertEqual(
                samples.var(dim=0) / total_count,
                bin1.variance / total_count,
                atol=0.02,
                rtol=0,
            )

    def test_dirichlet_shape(self):
        alpha = torch.randn(2, 3).exp().requires_grad_().to("xpu")
        alpha_1d = torch.randn(4).exp().requires_grad_().to("xpu")
        self.assertEqual(Dirichlet(alpha).sample().size(), (2, 3))
        self.assertEqual(Dirichlet(alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Dirichlet(alpha_1d).sample().size(), (4,))
        self.assertEqual(Dirichlet(alpha_1d).sample((1,)).size(), (1, 4))

    def test_dirichlet_shape2(self):
        dist = Dirichlet(torch.tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]]).to("xpu"))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4, 3, 2)))
        self.tensor_sample_1 = torch.ones(3, 2).to("xpu")
        self.tensor_sample_2 = torch.ones(3, 2, 3).to("xpu")
        simplex_sample = self.tensor_sample_1 / self.tensor_sample_1.sum(
            -1, keepdim=True
        )
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        simplex_sample = torch.ones(3, 1, 2).to("xpu")
        simplex_sample = simplex_sample / simplex_sample.sum(-1).unsqueeze(-1)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3, 3)))

    def test_dirichlet_mean_var(self):
        num_samples = 1000000
        alpha = torch.exp(torch.randn(3, dtype=torch.float, device=sycl_device))
        dist = torch.distributions.Dirichlet(alpha)
        samples_ret = dist.sample((num_samples,))

        alpha_sum = alpha.sum(dim=0)
        ref_mean = alpha / alpha_sum
        ref_var = (alpha * (alpha_sum - alpha)) / (
            alpha_sum * alpha_sum * (alpha_sum + 1)
        )
        self.assertEqual(samples_ret.mean(dim=0).cpu(), ref_mean, atol=5e-3, rtol=5e-3)
        self.assertEqual(samples_ret.var(dim=0).cpu(), ref_var, atol=5e-3, rtol=5e-3)

    def test_dirichlet_grad(self):
        concentration_cpu = torch.tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]])
        concentration_xpu = concentration_cpu.to("xpu")

        x_cpu = torch._sample_dirichlet(concentration_cpu)
        x_xpu = x_cpu.to("xpu")

        total_cpu = concentration_cpu.sum(-1, True).expand_as(concentration_cpu)
        total_xpu = total_cpu.to("xpu")

        grad_cpu = torch._dirichlet_grad(x_cpu, concentration_cpu, total_cpu)
        grad_xpu = torch._dirichlet_grad(x_xpu, concentration_xpu, total_xpu)
        self.assertEqual(grad_xpu.to("cpu"), grad_cpu)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_poisson_shape(self):
        set_rng_seed(1)
        rate = torch.randn(2, 3).abs().requires_grad_().to("xpu")
        rate_1d = torch.randn(1).abs().requires_grad_().to("xpu")
        self.assertEqual(Poisson(rate).sample().size(), (2, 3))
        self.assertEqual(Poisson(rate).sample((7,)).size(), (7, 2, 3))
        self.assertEqual(Poisson(rate_1d).sample().size(), (1,))
        self.assertEqual(Poisson(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Poisson(2.0).sample((2,)).size(), (2,))

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_poisson_mean_var(self):
        set_rng_seed(1)
        lamb = torch.randint(0, 5, [1]).float()
        lamb_arr = lamb.repeat(1000000000).to("xpu")
        dist = torch.distributions.Poisson(lamb_arr)
        sample_ret = dist.sample()
        self.assertEqual(
            torch.mean(sample_ret).cpu(), torch.mean(lamb).cpu(), atol=5e-4, rtol=5e-4
        )
        self.assertEqual(
            torch.var(sample_ret).cpu(), torch.mean(lamb).cpu(), atol=5e-4, rtol=5e-4
        )
