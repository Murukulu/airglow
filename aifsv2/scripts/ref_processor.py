"""PyTorch reference for the inside of the processor, one stage at a time.

Calls each TransformerProcessorBlock's submodules by hand in the order
TransformerProcessorBlock.forward and MultiHeadSelfAttention.attention_computation
(anemoi 0.9.3) run them, and writes data/dump/ref_proc<layer>_<name>.npy for the
same-named dumps transformer.rs makes: every layer's output (proc<layer>_out,
TransformerProcessorConfig dump_outputs) and, for the layers in --layers, every
stage of the block (dump_layers). Layers are counted across chunks, as in Rust.

The processor is fed Rust's own enc_x_latent dump, not PyTorch's, so the
comparison is of the processor alone: an encoder that is still wrong upstream
does not make every processor stage fail with it. Because that input is not
what the hooked run saw, the hand-run is separately checked by re-running it
from ref_enc_x_latent.npy and comparing to the hooked ref_proc_x_latent.npy,
so a mistake in this hand-run cannot masquerade as a Rust bug.

Attention is SDPA with the sliding window --window selects: the checkpoint's
own half-width by default (1120 for aifs-single-mse-2.0), `--window None` for
full attention, which is what transformer.rs computes until it windows, or any
integer. The self-check ignores --window and runs the hand-run both without and
with the checkpoint's window, saying which one the hooked run was: the hooked
proc_x_latent, dec_x_out and post references only pair with a Rust side in the
same mode, so regenerate them with ref_forward.py (--window or not) to match.

Needs the reference venv and a GPU; run the Rust binary with AIFS_DUMP_DIR=data/dump first (for
enc_x_latent) and ref_forward.py (for the end-to-end ref_proc_x_latent).
    python scripts/ref_processor.py [--layers 0 15] [--window None|1120|<int>]
"""

import argparse
import sys
from pathlib import Path

import einops
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_model import load_interface, use_sdpa  # noqa: E402
from ref_forward import DUMP_DIR, read_dump  # noqa: E402


def blocks(proc):
    """Every TransformerProcessorBlock in forward order, across the chunks."""
    return [block for chunk in proc.proc for block in chunk.blocks]


def run_block(block, x, ref, tag, intermediates):
    """One TransformerProcessorBlock.forward; records tag_out into ref (None records nothing),
    and every stage as well when intermediates is set."""
    rec = (
        (lambda name, t: ref.__setitem__(f"{tag}_{name}", t))
        if ref is not None and intermediates
        else (lambda *_: None)
    )
    attn = block.attention
    heads = attn.num_heads
    # transformer.rs has no q_norm/k_norm; if a checkpoint turned them on, this run would be
    # right and Rust wrong, for a reason no stage name would point at.
    assert not attn.qk_norm, (
        "processor attention uses qk_norm, which transformer.rs does not implement"
    )

    x_norm = block.layer_norm_attention(x)
    rec("x_norm", x_norm)
    query, key, value = attn.lin_q(x_norm), attn.lin_k(x_norm), attn.lin_v(x_norm)
    rec("query", query)
    rec("key", key)
    rec("value", value)
    # attention_computation: "(batch grid) (heads vars) -> batch heads grid vars", batch 1.
    q, k, v = (
        einops.rearrange(t, "(b g) (h d) -> b h g d", b=1, h=heads)
        for t in (query, key, value)
    )
    rec("query_rearranged", q)
    rec("key_rearranged", k)
    rec("value_rearranged", v)
    out = attn.attention(
        q,
        k,
        v,
        1,
        causal=False,
        window_size=attn.window_size,
        dropout_p=0.0,
        softcap=attn.softcap,
        alibi_slopes=attn.alibi_slopes,
    )
    out = einops.rearrange(out, "b h g d -> (b g) (h d)")
    rec("attn", out)
    proj = attn.projection(out)
    rec("proj", proj)
    x = x + proj
    rec("attn_res", x)

    mlp_norm = block.layer_norm_mlp(x)
    rec("mlp_norm", mlp_norm)
    mlp = block.mlp(mlp_norm)
    rec("mlp", mlp)
    x = x + mlp
    if ref is not None:
        ref[f"{tag}_out"] = x
    return x


def run_processor(proc, x, ref, layers=()):
    for layer, block in enumerate(blocks(proc)):
        x = run_block(block, x, ref, f"proc{layer}", layer in layers)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0],
        help="layers to dump every stage of, not just the output; match dump_layers in aifs.rs",
    )
    parser.add_argument(
        "--window",
        type=lambda s: None if s.lower() == "none" else int(s),
        default="checkpoint",
        help="attention half-window for the dumps: an integer, or None for full attention "
        "(default: the checkpoint's own)",
    )
    args = parser.parse_args()

    interface = load_interface()
    proc = interface.model.processor
    # The checkpoint's window, read before use_sdpa(window=False) clears it.
    window = {blk.attention.window_size for blk in blocks(proc)}
    assert len(window) == 1, f"blocks disagree on window_size: {window}"
    (window,) = window
    interface = use_sdpa(interface, window=False).cuda().eval()
    n = len(blocks(proc))
    assert all(0 <= l < n for l in args.layers), f"layers must be in 0..{n}"

    dump_window = window if args.window == "checkpoint" else args.window

    def set_window(size):
        for blk in blocks(proc):
            blk.attention.window_size = size
            # BandedSDPA caches its mask by sequence length alone, so a change of window at the
            # same length would silently reuse the old one.
            blk.attention.attention._mask = None

    with torch.inference_mode():
        # The hand-run has to reproduce the hooked run before it can judge Rust. The hooked
        # ref_proc_x_latent includes the latent skip that predict_step adds outside the processor.
        # Both window settings are tried so a hooked run made with --window is named rather than
        # just reported as a mismatch.
        hooked_in, hooked_out = (
            DUMP_DIR / "ref_enc_x_latent.npy",
            DUMP_DIR / "ref_proc_x_latent.npy",
        )
        if hooked_in.exists() and hooked_out.exists():
            x_in = torch.from_numpy(np.load(hooked_in)).cuda()
            b = np.load(hooked_out)
            for label, size in (("no window", None), (f"window {window}", window)):
                set_window(size)
                a = (run_processor(proc, x_in, None) + x_in).cpu().numpy()
                rel = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2))
                print(
                    f"hand-run ({label}) vs hooked proc_x_latent: rel rms {rel:.3e} "
                    f"({'ok' if rel < 1e-4 else 'MISMATCH'})"
                )
        else:
            print(
                "no ref_enc_x_latent / ref_proc_x_latent: run ref_forward.py to self-check the hand-run"
            )

        set_window(dump_window)
        print(f"dumping with window {dump_window} ({'full attention' if dump_window is None else 'half-width'})")
        ref = {}
        x = read_dump("enc_x_latent").cuda()
        run_processor(proc, x, ref, layers=set(args.layers))

    for name, t in ref.items():
        np.save(DUMP_DIR / f"ref_{name}.npy", t.float().cpu().numpy())
        print(f"ref_{name} {tuple(t.shape)}")


if __name__ == "__main__":
    main()
