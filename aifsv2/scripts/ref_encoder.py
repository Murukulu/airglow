"""PyTorch reference for the inside of the encoder, one stage at a time.

anemoi runs the mapper with edge sharding: destination nodes in chunks, source
nodes dropped and relabelled per chunk, so forward hooks on the submodules fire
several times on relabelled subsets and pair with nothing on the Rust side.
Instead this calls the encoder's submodules by hand on the full graph, in the
order GraphTransformerForwardMapper and GraphTransformerMapperBlock (anemoi
0.9.3) run them, and writes each result as data/dump/ref_<name>.npy for the
same-named dumps in encoder.rs / block.rs. The last stage is recomputed
enc_x_latent and is checked against the hooked run's ref_enc_x_latent.npy, so
a mistake in this hand-run cannot masquerade as a Rust bug.

It also checks the graph the encoder holds against the graph the Rust side
loads: the edge index and the first three edge attributes must be identical.

Needs the reference venv and a GPU; run ref_forward.py first (for x and the
end-to-end ref_enc_x_latent).
    python scripts/ref_encoder.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_model import load_interface, use_sdpa  # noqa: E402
from ref_forward import DUMP_DIR, read_dump  # noqa: E402

GRAPH = Path(__file__).resolve().parent.parent / "data" / "aifs-single-mse-2.0_graph.safetensors"


def check_graph(enc) -> None:
    graph = load_file(str(GRAPH))
    edge_index = enc.edge_index_base.cpu().numpy()
    same_index = np.array_equal(edge_index, graph["data_to_hidden.edge_index"])
    # sub_graph_edge_attributes: [edge_length, edge_dirs] in this checkpoint's config.
    attrs = np.concatenate([graph["data_to_hidden.edge_length"], graph["data_to_hidden.edge_dirs"]], 1)
    same_attr = np.allclose(enc.edge_attr.cpu().numpy(), attrs, atol=1e-7)
    print(f"edge_index {tuple(edge_index.shape)} == graph file: {same_index}")
    print(f"edge_attr (length, dirs) == graph file: {same_attr}")


def main() -> None:
    interface = use_sdpa(load_interface(), window=False).cuda().eval()
    m = interface.model
    enc, blk = m.encoder, m.encoder.proc
    check_graph(enc)

    x = read_dump("x").cuda()
    with torch.inference_mode():
        # Up to the encoder's door, exactly as predict_step gets there.
        x = interface.pre_processors(x[:, :, None, ...], in_place=False)
        x_data_latent, _, _ = m._assemble_input(x, batch_size=1)
        x_hidden_latent = m.node_attributes(m._graph_name_hidden, batch_size=1)

        ref = {}
        # GraphTransformerForwardMapper.pre_process
        ref["enc_x_src_emb"] = x_src = enc.emb_nodes_src(x_data_latent)
        ref["enc_x_dst_emb"] = x_dst = enc.emb_nodes_dst(x_hidden_latent)
        # prepare_edges: [edges, 1 length + 2 dirs + 8 trainable]; lin_edge reads this directly.
        ref["enc_edge_attr"] = edge_attr = enc.trainable(enc.edge_attr, batch_size=1)
        edge_index = enc._expand_edges(enc.edge_index_base, enc.edge_inc, batch_size=1)

        # GraphTransformerMapperBlock.forward
        ref["enc_x_src_norm"] = x_src_n = blk.layer_norm_attention_src(x_src)
        ref["enc_x_dst_norm"] = x_dst_n = blk.layer_norm_attention_dest(x_dst)
        ref["enc_x_r"] = x_r = blk.lin_self(x_dst_n)
        ref["enc_query"] = query = blk.lin_query(x_dst_n)
        ref["enc_key"] = key = blk.lin_key(x_src_n)
        ref["enc_value"] = value = blk.lin_value(x_src_n)
        ref["enc_edges"] = edges = blk.lin_edge(edge_attr)
        heads, per_head = blk.num_heads, blk.out_channels_conv
        split = lambda t: t.view(-1, heads, per_head)  # "(heads vars)": head is the outer index
        out = blk.conv(
            query=split(query), key=split(key), value=split(value), edge_attr=split(edges),
            edge_index=edge_index, size=(x_src.shape[0], x_dst.shape[0]),
        )
        ref["enc_conv"] = out = out.reshape(-1, heads * per_head)
        ref["enc_proj"] = out = blk.projection(out + x_r)
        ref["enc_attn_out"] = out = out + x_dst
        ref["enc_x_latent_hand"] = blk.node_dst_mlp(blk.layer_norm_mlp_dst(out)) + out

    for name, t in ref.items():
        np.save(DUMP_DIR / f"ref_{name}.npy", t.float().cpu().numpy())
        print(f"ref_{name} {tuple(t.shape)}")

    # The hand-run has to reproduce the hooked run before it can judge Rust.
    hooked = DUMP_DIR / "ref_enc_x_latent.npy"
    if hooked.exists():
        a, b = ref["enc_x_latent_hand"].cpu().numpy(), np.load(hooked)
        rel = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2))
        print(f"hand-run vs hooked enc_x_latent: rel rms {rel:.3e} ({'ok' if rel < 1e-4 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
