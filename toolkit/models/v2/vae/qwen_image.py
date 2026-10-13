from contextlib import contextmanager

import torch
from diffusers import AutoencoderKLQwenImage

from .._mixin import OstrisModelMixin


class QwenImageVAE(AutoencoderKLQwenImage, OstrisModelMixin):
    """The Qwen-Image (wan-style video) VAE, shared by the qwen_image family,
    nucleus_image and krea2."""

    aitk_subfolder = "vae"


class QwenImageVAEHolderMixin:
    """BaseModel-side encode_images/decode_latents for models whose self.vae
    is the Qwen-Image VAE: it is a video VAE, so images ride in a single-frame
    dim and latents are normalized with the config's latents_mean/std."""

    # tile the decode when low_vram (encode tiling is opt-in via vae_tiling)
    vae_decode_tiled_on_low_vram = False

    _VAE_TILING_ATTRS = (
        "use_tiling",
        "tile_sample_min_height",
        "tile_sample_min_width",
        "tile_sample_stride_height",
        "tile_sample_stride_width",
    )

    @contextmanager
    def _vae_encode_tiling(self):
        """Enable the VAE's native tiled encode (tiles run one at a time, overlaps
        are linearly blended on the latent moments) when model_config.vae_tiling
        is set. The VAE's previous tiling state is restored on exit, since
        enable_tiling() also overwrites the tile sizes the tiled decode uses."""
        if not getattr(self.model_config, "vae_tiling", False):
            yield
            return
        saved = {name: getattr(self.vae, name) for name in self._VAE_TILING_ATTRS}
        tile = self.model_config.vae_tile_size
        stride = tile - self.model_config.vae_tile_overlap
        self.vae.enable_tiling(
            tile_sample_min_height=tile,
            tile_sample_min_width=tile,
            tile_sample_stride_height=stride,
            tile_sample_stride_width=stride,
        )
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(self.vae, name, value)

    def encode_images(self, image_list, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        images = images.unsqueeze(2)  # add the frame dim
        with self._vae_encode_tiling():
            latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.squeeze(2)  # drop the frame dim
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents = latents.unsqueeze(2)  # add the frame dim

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        # full-resolution decode spikes VRAM; models opt in to tiling it
        tiled = self.vae_decode_tiled_on_low_vram and self.model_config.low_vram
        if tiled:
            self.vae.enable_tiling()
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()

        images = images.squeeze(2)  # drop the frame dim
        return images.to(device, dtype=dtype)
