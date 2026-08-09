"""Convert an Anemoi/Lightning .ckpt to safetensors.

Works on macOS without flash-attn or the full anemoi stack installed.

THE PROBLEM
-----------
torch.load uses pickle to reconstruct the model object. Pickle needs to import
every class referenced in the file — including
anemoi.models.layers.chunk.TransformerProcessorChunk (removed in newer
anemoi-models) and flash_attn.flash_attn_interface.flash_attn_func (doesn't
compile on macOS). The load fails before reading a single weight.

THE FIX
-------
We swap in a custom unpickler (_StubUnpickler) that catches any
ImportError/ModuleNotFoundError in find_class and returns a dummy _Stub class
instead. PyTorch's torch.load accepts a pickle_module argument, so we pass our
patched one. The real tensor data (stored as raw bytes in the zip, not pickled)
loads fine — only the Python object graph is affected by the stub.

WHY _Stub MUST SUBCLASS nn.Module
----------------------------------
The checkpoint is an inference format, not a Lightning training checkpoint. The
top-level object is AnemoiModelInterface, which is a real nn.Module that owns
sub-modules. When we call .state_dict() on it, PyTorch traverses _modules
recursively. If the stubs aren't nn.Module subclasses, they don't register
their _parameters/_buffers, so all 255M weights get silently dropped.

WHY .clone()
------------
The normalizer tensors (_norm_mul, _norm_add, etc.) are registered as shared
storage — pre_processors and post_processors point to the same underlying
memory. safetensors refuses to save aliased tensors because it can't round-trip
them correctly. Cloning gives each key its own independent storage.

OUTPUT
------
290 tensors: encoder/decoder weights, 208 processor layer weights, normalizer
affine params (pre/post processors), and node lat/lon attributes.

Handles two checkpoint formats:
  - Inference .ckpt: top-level object is AnemoiModelInterface (nn.Module)
  - Lightning training .ckpt: top-level is a dict with a "state_dict" key

Usage:
    uv run python ckpt_to_safetensors.py <path/to/model.ckpt> [output.safetensors]

Burn note: PyTorch Linear weights are [out, in]; Burn's are [in, out].
Use PyTorchToBurnAdapter (burn-store) or transpose manually on load.
"""

import argparse
import pickle
import zipfile
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file


class _Stub(nn.Module):
    """Placeholder for any missing anemoi/flash-attn class during unpickling.

    Must subclass nn.Module so that state_dict() traversal still works when the
    inference checkpoint is an AnemoiModelInterface that owns Stub sub-modules.
    """

    def __init__(self, *a, **kw):
        super().__init__()

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __call__(self, *a, **kw):
        return _Stub()


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            # i.e. when flash-attn fails.
            return type(name, (_Stub,), {"__module__": mod})


class _stub_pickle_module:
    Unpickler = _StubUnpickler

    @staticmethod
    def load(f, **kw):
        return _StubUnpickler(f).load()


def load_ckpt(ckpt_path: Path) -> dict:
    return torch.load(
        ckpt_path,
        map_location="cpu",
        pickle_module=_stub_pickle_module,
        weights_only=False,
    )


def extract_state_dict(ckpt) -> dict[str, torch.Tensor]:
    # Lightning training checkpoint: top-level is a dict
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt)
        tensors = {k: v.clone().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    # Inference checkpoint: top-level is an nn.Module (AnemoiModelInterface)
    elif isinstance(ckpt, nn.Module):
        tensors = {k: v.clone().contiguous() for k, v in ckpt.state_dict().items()}
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    if not tensors:
        raise ValueError("No tensors found — check the checkpoint structure.")
    return tensors


def strip_prefix(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in tensors.items()}


def print_summary(tensors: dict[str, torch.Tensor]) -> None:
    total = sum(v.numel() for v in tensors.values())
    print(f"  {len(tensors)} tensors, {total / 1e6:.1f}M parameters")
    for k, v in list(tensors.items())[:10]:
        print(f"  {k:60s} {str(list(v.shape)):20s} {v.dtype}")
    if len(tensors) > 10:
        print(f"  ... and {len(tensors) - 10} more")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", type=Path, help="Input .ckpt file")
    parser.add_argument("output", type=Path, nargs="?", help="Output .safetensors file (default: same stem)")
    parser.add_argument("--strip-prefix", default="", metavar="PREFIX",
                        help="Strip key prefix before saving (e.g. 'model.' for Lightning training ckpts)")
    parser.add_argument("--no-strip", action="store_true", help="Keep keys exactly as-is")
    parser.add_argument("--list-keys", action="store_true", help="Print all keys and exit")
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt
    out_path: Path = args.output or ckpt_path.with_suffix(".safetensors")

    print(f"Loading {ckpt_path} ...")
    ckpt = load_ckpt(ckpt_path)

    if isinstance(ckpt, dict):
        top_keys = [k for k in ckpt if not isinstance(ckpt[k], torch.Tensor)]
        print(f"  Format: Lightning training checkpoint, top-level keys: {top_keys}")
    else:
        print(f"  Format: Inference checkpoint ({type(ckpt).__name__})")
        print(f"  Modules: {list(ckpt._modules.keys())}")

    tensors = extract_state_dict(ckpt)

    if args.list_keys:
        for k, v in sorted(tensors.items()):
            print(f"  {k:60s} {list(v.shape)}")
        return

    if not args.no_strip and args.strip_prefix:
        old_keys = set(tensors)
        tensors = strip_prefix(tensors, args.strip_prefix)
        n_stripped = sum(1 for k in old_keys if k.startswith(args.strip_prefix))
        print(f"  Stripped prefix '{args.strip_prefix}' from {n_stripped}/{len(old_keys)} keys")

    print("State dict summary:")
    print_summary(tensors)

    # Print non-tensor metadata from the checkpoint (useful for Burn: variable list, norm stats)
    if isinstance(ckpt, dict) and "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
        hp = ckpt["hyper_parameters"]
        print(f"\n  hyper_parameters keys: {list(hp.keys())[:15]}")
    elif isinstance(ckpt, nn.Module) and hasattr(ckpt, "config"):
        print(f"\n  config type: {type(ckpt.config).__name__}")

    print(f"\nSaving to {out_path} ...")
    save_file(tensors, out_path)
    print("Done.")

    # Also dump metadata JSON from the zip for variable ordering / norm stats
    meta_out = out_path.with_name(out_path.stem + "_metadata.json")
    try:
        with zipfile.ZipFile(ckpt_path) as zf:
            json_files = [n for n in zf.namelist() if n.endswith(".json")]
            if json_files:
                import json
                all_meta = {}
                for jf in json_files:
                    all_meta[jf] = json.loads(zf.read(jf))
                meta_out.write_text(json.dumps(all_meta, indent=2))
                print(f"Metadata JSON written to {meta_out}")
    except Exception as e:
        print(f"  (Could not extract metadata JSON: {e})")


if __name__ == "__main__":
    main()
