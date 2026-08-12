"""Extract the graph (HeteroData) from an Anemoi inference .ckpt to safetensors.

WHY
---
The .safetensors extract holds only the state_dict, which carries the 8 trainable
columns per edge but no connectivity. The .ckpt carries the built graph:
AnemoiModelInterface.graph_data is a pickled torch_geometric HeteroData with
edge_index, edge_length and edge_dirs for both bipartite sub-graphs, plus the raw
[N, 2] node coordinates. Dumping it here means the Rust runtime never has to
rebuild the graph geometrically (no kd-tree, no haversine, no Rodrigues, no
unit-std normalisation) — edge_length and edge_dirs are stored already normalised.

Unpickling reuses ckpt_to_safetensors._StubUnpickler, so missing classes
(flash_attn, renamed anemoi layers) are stubbed rather than fatal. torch_geometric
must be importable for real: HeteroData is what we are reading.

EDGE INDEX DTYPE
----------------
PyG stores edge_index as int64. Burn's Int element is i32 on most backends, and
the largest node id here is 542,079, so the default is to narrow to int32 after
asserting it fits. Pass --edge-index-dtype int64 to keep it wide.

Usage:
    .venv/bin/python scripts/extract_graph.py data/aifs-single-mse-2.0.ckpt
    .venv/bin/python scripts/extract_graph.py data/aifs-single-mse-2.0.ckpt --list
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_to_safetensors import load_ckpt  # noqa: E402


def find_graph(ckpt):
    """The graph hangs off AnemoiModelInterface.graph_data; a training ckpt keeps
    it in the top-level dict or under hyper_parameters."""
    if hasattr(ckpt, "graph_data"):
        return ckpt.graph_data
    if isinstance(ckpt, dict):
        for key in ("graph_data", "graph"):
            if key in ckpt:
                return ckpt[key]
        hp = ckpt.get("hyper_parameters")
        if isinstance(hp, dict) and "graph_data" in hp:
            return hp["graph_data"]
    raise ValueError(f"No graph_data on {type(ckpt).__name__}")


def describe(graph) -> None:
    print(f"  {type(graph).__name__}")
    for nt in graph.node_types:
        store = graph[nt]
        items = ", ".join(f"{k}={list(v.shape)}" for k, v in store.items() if torch.is_tensor(v))
        extra = ", ".join(f"{k}={v!r}" for k, v in store.items() if not torch.is_tensor(v))
        print(f"    node  {nt:8s} {items}  {extra}")
    for et in graph.edge_types:
        store = graph[et]
        items = ", ".join(f"{k}={list(v.shape)}" for k, v in store.items() if torch.is_tensor(v))
        extra = ", ".join(f"{k}={v!r}" for k, v in store.items() if not torch.is_tensor(v))
        src, _, dst = et
        print(f"    edge  {src}->{dst:8s} {items}  {extra}")


def flatten(graph, edge_index_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """HeteroData -> flat {name: tensor}. Node stores keep their type name, edge
    stores are keyed '{src}_to_{dst}' — so this survives a checkpoint whose node
    types are named differently, rather than hardcoding data/hidden."""
    out: dict[str, torch.Tensor] = {}

    def put(prefix: str, store) -> None:
        for key, value in store.items():
            if not torch.is_tensor(value):
                continue
            if key == "edge_index":
                limit = torch.iinfo(edge_index_dtype).max
                assert value.max().item() <= limit, f"{prefix}.edge_index exceeds {edge_index_dtype}"
                value = value.to(edge_index_dtype)
            # clone: PyG slices share storage, which safetensors refuses to save.
            out[f"{prefix}.{key}"] = value.clone().contiguous()

    for nt in graph.node_types:
        put(nt, graph[nt])
    for src, _, dst in graph.edge_types:
        put(f"{src}_to_{dst}", graph[(src, "to", dst)])
    return out


def check(tensors: dict[str, torch.Tensor]) -> None:
    """The oracle from the pipeline doc: the per-edge trainable tensors in the
    state_dict pin E exactly, and both edge attributes are stored unit-std."""
    for name, tensor in sorted(tensors.items()):
        if name.endswith(".edge_index"):
            print(f"  {name:34s} {list(tensor.shape)} {tensor.dtype}  E={tensor.shape[1]}")
        elif name.endswith((".edge_length", ".edge_dirs")):
            std = tensor.std().item()  # unbiased, over the flattened tensor
            flag = "ok" if abs(std - 1.0) < 1e-5 else "NOT unit-std"
            print(f"  {name:34s} {list(tensor.shape)} {tensor.dtype}  std={std:.7f} {flag}")
        else:
            print(f"  {name:34s} {list(tensor.shape)} {tensor.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ckpt", type=Path, help="Input .ckpt file")
    parser.add_argument("output", type=Path, nargs="?", help="Default: <ckpt stem>_graph.safetensors")
    parser.add_argument("--list", action="store_true", help="Print the HeteroData structure and exit")
    parser.add_argument("--edge-index-dtype", choices=("int32", "int64"), default="int32")
    args = parser.parse_args()

    out_path = args.output or args.ckpt.with_name(args.ckpt.stem + "_graph.safetensors")

    print(f"Loading {args.ckpt} ...")
    graph = find_graph(load_ckpt(args.ckpt))
    describe(graph)

    if args.list:
        return

    tensors = flatten(graph, getattr(torch, args.edge_index_dtype))
    print(f"\n{len(tensors)} tensors:")
    check(tensors)

    save_file(tensors, out_path)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"\nWrote {out_path} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
