"""Synthetic tests for krea2 sample loras (lora pairs, LyCORIS LoKr, full diffs) and
their activate/deactivate lifecycle. Tiny in-memory modules and temp safetensors only.

    python -m unittest testing.test_krea2_sample_loras -v
"""
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from extensions_built_in.diffusion_models.krea2.krea2 import (
    Krea2Model,
    _build_sample_lora_delta,
    _diffusers_to_krea2_name,
)
from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = torch.nn.Linear(12, 8, bias=False)  # LoKr: (2x3) kron (4x4)
        self.gate = torch.nn.Linear(12, 12, bias=False)


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.blocks[0].attn = Attn()
        self.txtfusion = torch.nn.Module()
        self.txtfusion.projector = torch.nn.Linear(12, 1, bias=False)

    def forward(self, x):
        a = self.blocks[0].attn
        return torch.cat([a.wk(x), a.gate(x), self.txtfusion.projector(x)], dim=-1)


def fake_model(transformer):
    m = Krea2Model.__new__(Krea2Model)
    m._sample_lora_handles = {}
    m.get_model_to_train = lambda: transformer
    m.print_and_status_update = print
    return m


class TestDeltaBuilders(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, 12)

    def test_lokr_full_factors_match_kron(self):
        w1, w2 = torch.randn(2, 3), torch.randn(4, 4)
        fn, fmt, shape = _build_sample_lora_delta(
            {"lokr_w1": w1, "lokr_w2": w2, "alpha": torch.tensor(16.0)}
        )
        self.assertEqual((fmt, shape), ("lokr", (8, 12)))
        # no decomposed factor -> alpha ignored, scale 1 (ComfyUI convention)
        torch.testing.assert_close(fn(self.x), F.linear(self.x, torch.kron(w1, w2)))

    def test_lokr_decomposed_factor_scaled_by_alpha_over_rank(self):
        w1 = torch.randn(2, 3)
        w2_a, w2_b = torch.randn(4, 2), torch.randn(2, 4)
        fn, _, _ = _build_sample_lora_delta(
            {"lokr_w1": w1, "lokr_w2_a": w2_a, "lokr_w2_b": w2_b, "alpha": torch.tensor(1.0)}
        )
        expected = F.linear(self.x, torch.kron(w1, w2_a @ w2_b)) * (1.0 / 2)
        torch.testing.assert_close(fn(self.x), expected)

    def test_lokr_bf16(self):
        w1, w2 = torch.randn(2, 3), torch.randn(4, 4)
        fn, _, _ = _build_sample_lora_delta({"lokr_w1": w1, "lokr_w2": w2})
        out = fn(self.x.bfloat16())
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(
            out.float(), F.linear(self.x, torch.kron(w1, w2)), rtol=5e-2, atol=5e-2
        )

    def test_diff_with_bias(self):
        diff, diff_b = torch.randn(1, 12), torch.randn(1)
        fn, fmt, shape = _build_sample_lora_delta({"diff": diff, "diff_b": diff_b})
        self.assertEqual((fmt, shape), ("diff", (1, 12)))
        torch.testing.assert_close(fn(self.x), F.linear(self.x, diff, diff_b))

    def test_lora_pair_unchanged(self):
        down, up = torch.randn(4, 12), torch.randn(8, 4)
        fn, fmt, shape = _build_sample_lora_delta(
            {"down": down, "up": up, "alpha": torch.tensor(2.0)}
        )
        self.assertEqual((fmt, shape), ("lora", (8, 12)))
        torch.testing.assert_close(fn(self.x), F.linear(F.linear(self.x, down), up) * 0.5)

    def test_unusable_parts(self):
        self.assertIsNone(_build_sample_lora_delta({"alpha": torch.tensor(1.0)}))
        self.assertIsNone(
            _build_sample_lora_delta(
                {"lokr_w1": torch.randn(2, 3), "lokr_w2": torch.randn(4, 4),
                 "lokr_t2": torch.randn(1, 1, 1, 1)}
            )
        )


class TestSampleLoraLifecycle(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.lokr_path = os.path.join(self.tmp.name, "lokr.safetensors")
        save_file(
            {
                "diffusion_model.blocks.0.attn.wk.lokr_w1": torch.randn(2, 3),
                "diffusion_model.blocks.0.attn.wk.lokr_w2": torch.randn(4, 4),
                "diffusion_model.blocks.0.attn.wk.alpha": torch.tensor(8.0),
                "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.randn(3, 3),
                "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.randn(4, 4),
                "diffusion_model.blocks.0.attn.gate.alpha": torch.tensor(8.0),
            },
            self.lokr_path,
        )
        self.diff_path = os.path.join(self.tmp.name, "diff.safetensors")
        save_file({"diffusion_model.txtfusion.projector.diff": torch.randn(1, 12)}, self.diff_path)

    def tearDown(self):
        self.tmp.cleanup()

    def run_lifecycle(self, device):
        transformer = TinyTransformer().to(device)
        model = fake_model(transformer)
        x = torch.randn(2, 12, device=device)
        with torch.no_grad():
            base = transformer(x)
            for p in (self.lokr_path, self.diff_path):
                model._sample_lora_handles[p] = model._load_sample_lora_hooks(p)
            self.assertEqual(model._sample_lora_handles[self.lokr_path]["matched"], 2)
            self.assertEqual(model._sample_lora_handles[self.diff_path]["matched"], 1)
            # loaded but inactive (training): output untouched
            torch.testing.assert_close(transformer(x), base, rtol=0, atol=0)

            cfg = type("Cfg", (), {"loras": [type("L", (), {"path": p, "weight": 1.0})()
                                             for p in (self.lokr_path, self.diff_path)]})()
            self.assertTrue(model.prepare_sample_loras([cfg]))
            model.set_sample_loras_for_config(cfg)
            sampled = transformer(x)
            self.assertFalse(torch.allclose(sampled[:, :8], base[:, :8]))  # wk (lokr)
            self.assertFalse(torch.allclose(sampled[:, 20:], base[:, 20:]))  # projector (diff)

            model.end_sample_loras()
            torch.testing.assert_close(transformer(x), base, rtol=0, atol=0)
        return model

    def test_lifecycle_cpu(self):
        self.run_lifecycle("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_no_vram_kept_after_sampling(self):
        # the first cuda matmul allocates cuBLAS's one-off workspace; keep it out
        a = torch.randn(12, 12, device="cuda")
        (a @ a).sum().item()
        del a
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        model = self.run_lifecycle("cuda")
        torch.cuda.synchronize()
        # only the tiny transformer itself (moved in run_lifecycle) is held on the GPU
        transformer_bytes = sum(
            p.numel() * p.element_size() for p in model.get_model_to_train().parameters()
        )
        self.assertLessEqual(torch.cuda.memory_allocated() - before, transformer_bytes + 4096)



class TestDiffusersKeyNames(unittest.TestCase):
    # every module-name pattern of a diffusers-format krea2 lora
    PATTERNS = [
        f"transformer.{blk}.{i}.{leaf}"
        for blk in ("transformer_blocks", "text_fusion.layerwise_blocks",
                    "text_fusion.refiner_blocks")
        for i in (0, 1)
        for leaf in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate",
                     "attn.to_out.0", "ff.gate", "ff.up", "ff.down")
    ]

    def test_translation(self):
        self.assertEqual(
            _diffusers_to_krea2_name("transformer.transformer_blocks.3.attn.to_out.0"),
            "transformer.blocks.3.attn.wo",
        )
        self.assertEqual(
            _diffusers_to_krea2_name("transformer.text_fusion.refiner_blocks.1.ff.down"),
            "transformer.txtfusion.refiner_blocks.1.mlp.down",
        )
        # original names are left alone
        self.assertEqual(_diffusers_to_krea2_name("blocks.0.attn.wk"), "blocks.0.attn.wk")

    def test_diffusers_lora_hooks_every_module_of_real_krea2_layout(self):
        torch.manual_seed(0)
        transformer = SingleStreamDiT(SingleMMDiTConfig(
            features=64, tdim=32, txtdim=64, heads=4, multiplier=2, layers=2,
            patch=2, channels=4, txtlayers=3, txtheads=4, txtkvheads=4,
        ))
        modules = dict(transformer.named_modules())
        sd = {}
        for key in self.PATTERNS:
            target = modules[_diffusers_to_krea2_name(key)[len("transformer."):]]
            sd[f"{key}.lora_A.weight"] = torch.randn(4, target.in_features)
            sd[f"{key}.lora_B.weight"] = torch.randn(target.out_features, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "diffusers_lora.safetensors")
            save_file(sd, path)
            handle = fake_model(transformer)._load_sample_lora_hooks(path)
        self.assertEqual(handle["matched"], len(self.PATTERNS))


if __name__ == "__main__":
    unittest.main()
