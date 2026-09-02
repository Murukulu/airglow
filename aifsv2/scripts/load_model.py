"""Unpickle an Anemoi inference .ckpt and hand back the AnemoiModelEncProcDec.

The checkpoint's top-level object is AnemoiModelInterface, which owns the
pre/post processors and, under `.model`, the AnemoiModelEncProcDec itself
(node_attributes, encoder, processor, decoder, boundings). Unpickling goes
through ckpt_to_safetensors._StubUnpickler so missing classes (flash_attn,
renamed anemoi layers) stub out instead of aborting the load.

Import it:
    from load_model import load_model
    model = load_model()                      # AnemoiModelEncProcDec

To run a forward, anemoi-models must be installed for real and the processor's
attention swapped off flash-attn, which is stubbed:
    interface = use_sdpa(load_interface()).cuda()
    y = interface.predict_step(x)

Or poke at it in a REPL, where `model` and `interface` are left in scope:
    .venv/bin/python -i scripts/load_model.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_to_safetensors import load_ckpt  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "data" / "aifs-single-mse-2.0.ckpt"


def load_interface(ckpt_path: Path = DEFAULT_CKPT):
    """The whole AnemoiModelInterface — processors included."""
    return load_ckpt(Path(ckpt_path))


def load_model(ckpt_path: Path = DEFAULT_CKPT):
    """Just the AnemoiModelEncProcDec."""
    return load_interface(ckpt_path).model


class BandedSDPA(nn.Module):
    """Drop-in for anemoi's FlashAttentionWrapper on PyTorch's fused attention.

    (it is annoying to use flash-attn)

    Same call signature; q/k/v arrive as [batch, heads, seq, dim]. flash_attn's
    window_size=(w, w) attends the keys with |i - j| <= w, which is the mask built
    here. Not anemoi's own SDPAAttentionWrapper: that pins the MATH backend, which
    materialises the [heads, seq, seq] scores -- ~100 GB at 40,320 hidden nodes.
    """

    def __init__(self):
        super().__init__()
        self._mask = None

    def forward(self, query, key, value, batch_size, causal=False, window_size=None,
                dropout_p=0.0, softcap=None, alibi_slopes=None):
        # flash_attn treats softcap 0 as off; the checkpoint's config has softcap: 0.0.
        assert not softcap and alibi_slopes is None, "softcap/alibi not supported by SDPA"
        mask = None
        if window_size is not None:
            seq_len = query.shape[-2]
            if self._mask is None or self._mask.shape[0] != seq_len:
                i = torch.arange(seq_len, device=query.device)
                self._mask = (i[None, :] - i[:, None]).abs() <= window_size
            mask = self._mask
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=causal
            )


def use_sdpa(interface, window: bool = True):
    """Swap every processor attention onto BandedSDPA so a forward runs without flash-attn.
    window=False drops the 1120-node sliding window, matching what the Rust
    processor currently computes (see transformer.rs).
    """
    from anemoi.models.layers.attention import MultiHeadSelfAttention
    for m in interface.model.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m.attention = BandedSDPA()
            if not window:
                m.window_size = None
    return interface


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    interface = load_interface(path)
    model = interface.model
    print(f"{type(model).__module__}.{type(model).__name__}")
    print(f"  submodules: {list(model._modules.keys())}")
    print(f"  attributes: {list(model.__dict__)}")
    print(f"  attributes: {interface.pre_processors.processors.normalizer._input_idx}")
    print(f"  attributes: {interface.pre_processors.processors.normalizer._output_idx}")
    print(f"  attributes: {interface.pre_processors.processors.normalizer._norm_mul}")
    # print(f"data_indices: {model.data_indices}")
    # print(f"  statistics: {model.statistics}")
    # print(f"  graph_data: {model._graph_data}")
    # print(f"  graph_data: {model._graph_data.node_types}")
