"""Inspect a safetensors file: list tensor keys or query shapes/dtypes.

Usage:
    uv run scripts/parse_safetensors.py                          # summary
    uv run scripts/parse_safetensors.py --keys                   # list all keys
    uv run scripts/parse_safetensors.py --query "model.decoder.emb_nodes_dst.*"
    uv run scripts/parse_safetensors.py --query weight.norm bias.norm
"""

# /// script
# dependencies = ["safetensors", "numpy"]
# ///

import argparse
from fnmatch import fnmatch
from pathlib import Path
from pprint import pprint

from safetensors import safe_open

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "aifs-single-mse-2.0.safetensors"


def matches(key: str, pattern: str) -> bool:
    return fnmatch(key, pattern) if any(c in pattern for c in "*?[") else pattern in key


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_PATH, help="Path to .safetensors file")
    parser.add_argument("--keys", "-k", action="store_true", help="Print all tensor keys and exit")
    parser.add_argument("--query", "-q", nargs="+", metavar="PATTERN",
                        help="Key names or glob patterns (e.g. 'model.decoder.*'); prints shape/dtype for matches")
    args = parser.parse_args()

    with safe_open(args.path, framework="numpy") as f:
        keys = sorted(f.keys())

        if args.keys:
            pprint(keys)
            return

        if args.query:
            for pattern in args.query:
                found = [k for k in keys if matches(k, pattern)]
                if not found:
                    print(f"  (no keys matched {pattern!r})")
                    continue
                info = {}
                for k in found:
                    s = f.get_slice(k)
                    info[k] = {"shape": s.get_shape(), "dtype": str(s.get_dtype())}
                pprint(info)
            return

        print(f"{args.path}")
        print(f"  {len(keys)} tensors")
        for k in keys[:10]:
            s = f.get_slice(k)
            print(f"  {k:60s} {str(s.get_shape()):20s} {s.get_dtype()}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")


if __name__ == "__main__":
    main()
