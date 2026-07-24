"""Fast, offline diagnostic for krea2 sample-lora attachment + key format.

Builds the SingleStreamDiT transformer on the `meta` device (instant: no weights,
no quantization, no VRAM, no 20-minute model load) and reproduces exactly how a
sample lora would attach + which key mapping matches. Run it in seconds:

    python scripts/diag_sample_lora.py /path/to/your_lora.safetensors

Paste the whole output. It tells us (a) whether the lora attaches to modules and
with which strategy, (b) the module names ai-toolkit expects, and (c) the keys
your lora file actually has — everything needed to fix the mapping without a run.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from extensions_built_in.diffusion_models.krea2.krea2 import KREA2_MMDIT_CONFIG  # noqa: E402
from extensions_built_in.diffusion_models.krea2.src.mmdit import (  # noqa: E402
    SingleStreamDiT,
    SingleMMDiTConfig,
)
from toolkit.lora_special import LoRASpecialNetwork, LINEAR_MODULES  # noqa: E402


def main():
    lora_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("=== building SingleStreamDiT on meta (instant) ===")
    cfg = SingleMMDiTConfig(**KREA2_MMDIT_CONFIG)
    with torch.device("meta"):
        tr = SingleStreamDiT(cfg)
    print("root class:", tr.__class__.__name__)

    # 1) structure: which linear modules exist and how they're named
    linear_present, sample_names = {}, []
    for nm, m in tr.named_modules():
        cn = m.__class__.__name__
        if cn in LINEAR_MODULES:
            linear_present[cn] = linear_present.get(cn, 0) + 1
            if len(sample_names) < 15:
                sample_names.append(nm)
    print("linear modules present:", linear_present)
    print("sample linear module names:")
    for n in sample_names:
        print("   ", n)

    # 2) attach matrix: target x transformer_only
    def build(target, tonly):
        net = LoRASpecialNetwork(
            text_encoder=None,
            unet=tr,
            lora_dim=16,
            multiplier=0.0,
            alpha=16,
            train_unet=True,
            train_text_encoder=False,
            network_type="lora",
            transformer_only=tonly,
            is_transformer=True,
            base_model=None,
            target_lin_modules=target,
            is_assistant_adapter=True,
        )
        return net

    print("\n=== attach matrix ===")
    good_net = None
    for target in (["SingleStreamDiT"], [tr.__class__.__name__]):
        for tonly in (True, False):
            try:
                net = build(target, tonly)
                n = len(getattr(net, "unet_loras", []) or [])
                print(f"target={target} transformer_only={tonly} -> {n} modules")
                if n > 0 and good_net is None:
                    good_net = net
            except Exception as e:  # noqa: BLE001
                print(f"target={target} transformer_only={tonly} -> ERROR: {e}")

    # 3) keys ai-toolkit expects for a sample lora
    if good_net is not None:
        keys = list(good_net.state_dict().keys())
        print(f"\n=== network expects {len(keys)} keys, sample: ===")
        for k in keys[:15]:
            print("   ", k)

    # 4) keys your lora file actually has
    if lora_path:
        print(f"\n=== lora file: {os.path.basename(lora_path)} ===")
        try:
            sd = load_file(lora_path)
            print(f"{len(sd)} keys, sample:")
            for k in list(sd.keys())[:15]:
                print("   ", k, tuple(sd[k].shape))
        except Exception as e:  # noqa: BLE001
            print("failed to read lora file:", e)
    else:
        print("\n(pass a lora path as arg to also dump its keys)")


if __name__ == "__main__":
    main()
