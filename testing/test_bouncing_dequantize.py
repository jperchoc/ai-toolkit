"""Synthetic regression test: a failed dequantize in the offloaded (bouncing) linear
backward must not leak a still-quantized torchao weight into `grad @ w`
(which crashed with "Expecting input size ... divisible by block_size").

    python -m unittest testing.test_bouncing_dequantize -v
"""
import unittest

import torch

from toolkit.memory_management import manager_modules as mm


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestBouncingDequantize(unittest.TestCase):
    def setUp(self):
        from torchao.dtypes import AffineQuantizedTensor
        from torchao.quantization.quant_api import Float8WeightOnlyConfig, quantize_

        self.aqt = AffineQuantizedTensor
        self.orig_dequantize = AffineQuantizedTensor.dequantize
        self.dev = torch.device("cuda")
        torch.manual_seed(0)
        # non-square like the krea2 mlp down proj (16384 -> 6144), scaled down
        lin = torch.nn.Linear(256, 96, bias=False).to(self.dev, torch.bfloat16)
        self.ref_weight = lin.weight.detach().clone()
        quantize_(lin, Float8WeightOnlyConfig())
        self.ref_weight = lin.weight.dequantize().detach().clone()
        mm._move_params_to_cpu_and_pin(lin)
        self.weight = lin.weight

    def tearDown(self):
        self.aqt.dequantize = self.orig_dequantize

    def fail_first_dequantizes(self, n, exc):
        calls = [0]
        orig = self.orig_dequantize

        def patched(t, *a, **k):
            calls[0] += 1
            if calls[0] <= n:
                raise exc
            return orig(t, *a, **k)

        self.aqt.dequantize = patched

    def forward(self):
        x = torch.randn(1, 8, 256, device=self.dev, dtype=torch.bfloat16, requires_grad=True)
        y = mm._BouncingLinearFn.apply(x, self.weight, None, self.dev)
        return x, y

    def test_transient_oom_in_backward_is_retried(self):
        x, y = self.forward()
        self.fail_first_dequantizes(1, torch.OutOfMemoryError("simulated"))
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        torch.testing.assert_close(x.grad, grad_out @ self.ref_weight)

    def test_persistent_oom_surfaces_as_oom(self):
        x, y = self.forward()
        self.fail_first_dequantizes(10, torch.OutOfMemoryError("simulated"))
        with self.assertRaises(torch.OutOfMemoryError):
            y.float().sum().backward()

    def test_other_failure_never_returns_quantized_weight(self):
        x, y = self.forward()
        self.fail_first_dequantizes(10, ValueError("simulated"))
        with self.assertRaisesRegex(RuntimeError, "could not dequantize"):
            y.float().sum().backward()


if __name__ == "__main__":
    unittest.main()
