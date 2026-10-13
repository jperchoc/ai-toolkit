"""Synthetic tests for vae_tiling on the Qwen-Image VAE encode (krea2, qwen_image,
nucleus_image). Uses a randomly initialized VAE and in-memory random tensors only.

    python -m unittest testing.test_vae_encode_tiling -v
"""
import unittest
from unittest import mock

import torch

from toolkit.config_modules import ModelConfig
from toolkit.models.v2.vae.qwen_image import QwenImageVAE, QwenImageVAEHolderMixin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16


class FakeQwenVAEModel(QwenImageVAEHolderMixin):
    def __init__(self, vae, **config_kwargs):
        self.vae = vae
        self.model_config = ModelConfig(name_or_path="dummy", arch="krea2", **config_kwargs)
        self.vae_device_torch = DEVICE
        self.vae_torch_dtype = DTYPE


def reference_encode(vae, image):
    """The original (pre-patch) untiled encode_images path."""
    images = image.unsqueeze(0).to(DEVICE, dtype=DTYPE).unsqueeze(2)
    latents = vae.encode(images).latent_dist.sample()
    # verbatim from the pre-patch QwenImageVAEHolderMixin.encode_images
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * latents_std
    return latents.squeeze(2).to(DEVICE, dtype=DTYPE)


class TestVaeEncodeTiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        # default config == Qwen-Image VAE architecture (f8, z_dim 16), random weights
        cls.vae = QwenImageVAE().to(DEVICE, dtype=DTYPE).eval().requires_grad_(False)

    def image(self, size=1024):
        g = torch.Generator().manual_seed(1)
        return torch.rand(3, size, size, generator=g) * 2 - 1  # [-1, 1] like the dataloader

    def tiling_state(self):
        return {n: getattr(self.vae, n) for n in QwenImageVAEHolderMixin._VAE_TILING_ATTRS}

    def test_tiled_geometry_and_bf16(self):
        model = FakeQwenVAEModel(self.vae, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=64)
        with torch.no_grad():
            latents = model.encode_images([self.image(1024)])
        self.assertEqual(tuple(latents.shape), (1, 16, 128, 128))
        self.assertEqual(latents.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(latents.float()).all())

    def test_non_square_geometry(self):
        model = FakeQwenVAEModel(self.vae, vae_tiling=True, vae_tile_size=256, vae_tile_overlap=64)
        img = torch.rand(3, 640, 1040) * 2 - 1  # not a multiple of the stride
        with torch.no_grad():
            latents = model.encode_images([img])
        self.assertEqual(tuple(latents.shape), (1, 16, 80, 130))

    def test_disabled_keeps_original_path(self):
        model = FakeQwenVAEModel(self.vae)  # vae_tiling absent
        self.assertFalse(model.model_config.vae_tiling)
        img = self.image(256)
        before = self.tiling_state()
        with torch.no_grad(), mock.patch.object(
            self.vae, "tiled_encode", wraps=self.vae.tiled_encode
        ) as spy, mock.patch.object(self.vae, "enable_tiling", wraps=self.vae.enable_tiling) as enable:
            torch.manual_seed(123)
            out = model.encode_images([img])
            torch.manual_seed(123)
            ref = reference_encode(self.vae, img)
        spy.assert_not_called()
        enable.assert_not_called()
        self.assertEqual(self.tiling_state(), before)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_tiles_run_one_at_a_time(self):
        model = FakeQwenVAEModel(self.vae, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=64)
        in_flight, max_in_flight, shapes = [0], [0], []

        def pre_hook(_module, args, _kwargs):
            in_flight[0] += 1
            max_in_flight[0] = max(max_in_flight[0], in_flight[0])
            shapes.append(tuple(args[0].shape))

        def post_hook(_module, _args, _kwargs, _out):
            in_flight[0] -= 1

        h1 = self.vae.encoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
        h2 = self.vae.encoder.register_forward_hook(post_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                model.encode_images([self.image(1024)])
        finally:
            h1.remove()
            h2.remove()
        # stride 448 over 1024 px -> 3 x 3 tiles, each a single-image, <= 512 px chunk
        self.assertEqual(len(shapes), 9)
        self.assertEqual(max_in_flight[0], 1)
        for b, c, f, h, w in shapes:
            self.assertEqual((b, c, f), (1, 3, 1))
            self.assertLessEqual(h, 512)
            self.assertLessEqual(w, 512)

    def test_state_restored_after_encode(self):
        before = self.tiling_state()
        model = FakeQwenVAEModel(self.vae, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=64)
        with torch.no_grad():
            model.encode_images([self.image(768)])
        self.assertEqual(self.tiling_state(), before)
        with mock.patch.object(self.vae, "tiled_encode", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError), torch.no_grad():
                model.encode_images([self.image(768)])
        self.assertEqual(self.tiling_state(), before)

    def test_image_smaller_than_tile_is_identical_to_untiled(self):
        tiled = FakeQwenVAEModel(self.vae, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=64)
        plain = FakeQwenVAEModel(self.vae)
        img = self.image(512)
        with torch.no_grad():
            torch.manual_seed(7)
            a = tiled.encode_images([img])
            torch.manual_seed(7)
            b = plain.encode_images([img])
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_tiling_lowers_peak_vram(self):
        def peak(model):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.no_grad():
                model.encode_images([self.image(1024)])
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base

        tiled = peak(FakeQwenVAEModel(self.vae, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=64))
        try:
            untiled = peak(FakeQwenVAEModel(self.vae))
        except torch.OutOfMemoryError:
            untiled = None  # the untiled encode does not even fit on this GPU
        finally:
            torch.cuda.empty_cache()
        untiled_str = "OOM" if untiled is None else f"{untiled / 2**30:.2f} GiB"
        print(f"\n  1024x1024 encode peak: untiled {untiled_str}, tiled {tiled / 2**30:.2f} GiB")
        if untiled is not None:
            self.assertLess(tiled, untiled * 0.6)

    def test_config_validation(self):
        for bad in (
            dict(vae_tile_size=500),
            dict(vae_tile_size=0),
            dict(vae_tile_overlap=60),
            dict(vae_tile_overlap=-8),
            dict(vae_tile_size=256, vae_tile_overlap=256),
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                ModelConfig(name_or_path="dummy", arch="krea2", vae_tiling=True, **bad)
        # bad values are ignored while tiling is off
        ModelConfig(name_or_path="dummy", arch="krea2", vae_tile_size=500)


if __name__ == "__main__":
    unittest.main()
