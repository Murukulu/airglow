"""PyTorch reference for the inside of the decoder, one stage at a time.

Calls GraphTransformerBackwardMapper's submodules by hand on the full graph, in
the order GraphTransformerBackwardMapper and GraphTransformerMapperBlock (anemoi
0.9.3) run them, and writes data/dump/ref_dec_<name>.npy for the same-named
dumps decoder.rs / block.rs make. The message passing runs in destination-node
chunks (--chunks): the softmax and the sum are both per destination node, so the
split is exact, and it keeps the [edges, 1024] projected edge tensor -- 6.7 GB
for the 1,626,240 decoder edges -- off the GPU.

The decoder is fed Rust's own proc_x_latent and x_latent_data dumps, not
PyTorch's, so the comparison is of the decoder alone: the processor's drift
upstream does not make every decoder stage fail with it. ref_forward.py's hooked
capture is kept apart as ref_dec_x_out_hooked.npy and used two ways here:

  * the hand-run is re-run from PyTorch's own inputs and checked against it, so a
    mistake in this hand-run cannot masquerade as a Rust bug;
  * the hand-run on Rust's inputs is compared with the hand-run on PyTorch's
    inputs. No Rust code is involved in that number: it is how much the PyTorch
    decoder itself amplifies the difference between the two proc_x_latent
    tensors. If it is as large as the dec_x_out failure in compare_dump.py, the
    decoder is innocent and the fault is upstream; if it is small, the decoder
    is wrong and the staged dumps localise where.

It also checks the graph the decoder holds against the graph the Rust side
loads: the edge index and the first three edge attributes must be identical.

Needs the reference venv and a GPU; run the Rust binary with AIFS_DUMP_DIR=data/dump first (for
proc_x_latent and x_latent_data) and ref_forward.py (for the hooked capture).
    python scripts/ref_decoder.py [--chunks 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_model import load_interface, use_sdpa  # noqa: E402
from ref_forward import DUMP_DIR, read_dump  # noqa: E402

GRAPH = Path(__file__).resolve().parent.parent / "data" / "aifs-single-mse-2.0_graph.safetensors"


def check_graph(dec) -> None:
    graph = load_file(str(GRAPH))
    edge_index = dec.edge_index_base.cpu().numpy()
    same_index = np.array_equal(edge_index, graph["hidden_to_data.edge_index"])
    # sub_graph_edge_attributes: [edge_length, edge_dirs] in this checkpoint's config.
    attrs = np.concatenate([graph["hidden_to_data.edge_length"], graph["hidden_to_data.edge_dirs"]], 1)
    same_attr = np.allclose(dec.edge_attr.cpu().numpy(), attrs, atol=1e-7)
    print(f"edge_index {tuple(edge_index.shape)} == graph file: {same_index}")
    print(f"edge_attr (length, dirs) == graph file: {same_attr}")


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def run_decoder(dec, x_src, x_dst, chunks: int, rec=lambda name, t: t):
    """One GraphTransformerBackwardMapper.forward. x_src is the processor's latent
    [hidden, 1024], x_dst the assembled data input [data, in_channels_dst]. rec(name, t) sees each
    stage under the name decoder.rs dumps it as; returns the decoder output [data, out_channels]."""
    blk = dec.proc

    # prepare_edges / pre_process. The source side is passed through unembedded.
    edge_attr = rec("edge_attr", dec.trainable(dec.edge_attr, 1))  # [E, 1 + 2 + 8]
    src, dst = dec._expand_edges(dec.edge_index_base, dec.edge_inc, 1)
    x_dst = rec("x_dst_emb", dec.emb_nodes_dst(x_dst))

    # GraphTransformerMapperBlock.forward
    x_src_n = blk.layer_norm_attention_src(x_src)
    x_dst_n = blk.layer_norm_attention_dest(x_dst)
    x_r = blk.lin_self(x_dst_n)
    heads, per_head = blk.num_heads, blk.out_channels_conv
    split = lambda t: t.view(-1, heads, per_head)  # "(heads vars)": head is the outer index
    query = split(blk.lin_query(x_dst_n))
    key, value = split(blk.lin_key(x_src_n)), split(blk.lin_value(x_src_n))
    del x_src_n, x_dst_n

    n_src, n_dst = x_src.shape[0], x_dst.shape[0]
    out = torch.empty(n_dst, heads * per_head, device=x_dst.device, dtype=x_dst.dtype)
    for rows in torch.arange(n_dst, device=dst.device).tensor_split(chunks):
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        mask = (dst >= lo) & (dst < hi)
        edges = split(blk.lin_edge(edge_attr[mask]))
        conv = blk.conv(
            query=query[lo:hi], key=key, value=value, edge_attr=edges,
            edge_index=torch.stack([src[mask], dst[mask] - lo]), size=(n_src, hi - lo),
        )
        out[lo:hi] = conv.reshape(hi - lo, heads * per_head)
        del edges, conv, mask
    del query
    rec("conv", out)
    out = rec("proj", blk.projection(out + x_r))
    del x_r
    out = rec("attn_out", out + x_dst)
    del x_dst
    out = rec("block_out", blk.node_dst_mlp(blk.layer_norm_mlp_dst(out)) + out)

    # post_process: node_data_extractor is nn.Sequential(LayerNorm, Linear).
    norm, linear = dec.node_data_extractor
    out = rec("norm", norm(out))
    return rec("x_out", linear(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chunks", type=int, default=8,
                        help="destination-node chunks for the message passing; more is less GPU memory")
    args = parser.parse_args()

    interface = use_sdpa(load_interface(), window=False).cuda().eval()
    dec = interface.model.decoder
    check_graph(dec)

    with torch.inference_mode():
        # The hand-run has to reproduce the hooked run before it can judge Rust, and the same
        # hand-run on PyTorch's inputs is what the Rust-input run is measured against.
        hooked = DUMP_DIR / "ref_dec_x_out_hooked.npy"
        ref_in = [DUMP_DIR / "ref_proc_x_latent.npy", DUMP_DIR / "ref_x_latent_data.npy"]
        from_ref = None
        if hooked.exists() and all(p.exists() for p in ref_in):
            x_src, x_dst = (torch.from_numpy(np.load(p)).cuda() for p in ref_in)
            from_ref = run_decoder(dec, x_src, x_dst, args.chunks).cpu().numpy()
            del x_src, x_dst
            rel = rel_rms(from_ref, np.load(hooked))
            print(f"hand-run vs hooked dec_x_out: rel rms {rel:.3e} ({'ok' if rel < 1e-4 else 'MISMATCH'})")
        else:
            print("no ref_dec_x_out_hooked / ref_proc_x_latent / ref_x_latent_data: "
                  "run ref_forward.py to self-check the hand-run")

        def rec(name, t):
            # Written as it is produced, so the [data, 1024] stages do not pile up on the GPU.
            np.save(DUMP_DIR / f"ref_dec_{name}.npy", t.float().cpu().numpy())
            print(f"ref_dec_{name} {tuple(t.shape)}")
            return t

        x_src, x_dst = read_dump("proc_x_latent").cuda(), read_dump("x_latent_data").cuda()
        from_rust = run_decoder(dec, x_src, x_dst, args.chunks, rec).cpu().numpy()

    if from_ref is not None:
        # Pure PyTorch on both sides: the decoder's own gain on the upstream difference.
        latent = rel_rms(read_dump("proc_x_latent").numpy(), np.load(ref_in[0]))
        out = rel_rms(from_rust, from_ref)
        print(f"PyTorch decoder on Rust inputs vs on PyTorch inputs: rel rms {out:.3e} "
              f"(proc_x_latent inputs differ by {latent:.3e}, gain {out / latent:.1f}x)")
        print("  compare_dump.py's dec_x_out (Rust decoder on the same Rust inputs) should be "
              "far below this if the decoder is right")


if __name__ == "__main__":
    main()
