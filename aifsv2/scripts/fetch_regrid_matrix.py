"""Extract the 0.25° regular_ll -> N320 interpolation matrix to safetensors.

WHY
---
The forecast files are open data on a 1440 x 721 regular lat/lon grid (1,038,240 points);
the model runs on the N320 reduced Gaussian grid (542,080 points). Every retrieved field has
to cross that gap, and earthkit does not *compute* an interpolation — it applies a
precomputed sparse operator downloaded from ECMWF's matrix repository
(earthkit/regrid/interpolate.py:30-41, db.py:25). So the Rust runtime needs no interpolation
kernel, no stencil search and no Gaussian-latitude solver: it needs one CSR matrix and a
sparse matrix-vector product.

This script is the offline half. It resolves the matrix through earthkit's own index (rather
than a hardcoded hash, which would go stale), downloads and caches it, and rewrites the CSR
triple as safetensors — the format src/graph.rs already reads, so Rust needs no zip or .npy
reader.

The matrix is `mir` engine, version 16, method `linear`: 1,505,824 non-zeros over 542,080
rows, about 2.8 source points per target point.

COLUMN ORDER
------------
The matrix's input gridspec has area [90, 0, -90, 359.75] — column 0 is Greenwich. The GRIB
files in data/ start at the dateline (longitudeOfFirstGridPointInDegrees = 180), so the
caller must rotate each row by 180/0.25 = 720 columns before applying this. That rotation
lives in Rust, where the message's own keys are available; it is not baked in here.

Usage:
    data/aifs-single-mse-2.0/quiet_grub/.venv/bin/python scripts/fetch_regrid_matrix.py
    ... scripts/fetch_regrid_matrix.py --info          # resolve and describe, write nothing
    ... scripts/fetch_regrid_matrix.py --method nearest-neighbour --out other.safetensors

Requires earthkit-regrid (which brings scipy) and safetensors. The checkpoint's venv at
data/aifs-single-mse-2.0/quiet_grub/.venv has all three. Needs network on first run;
earthkit caches the download, so later runs are offline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from earthkit.regrid.db import SYS_DB
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "regrid-0p25-to-n320.safetensors"

# The two grids, in earthkit's gridspec vocabulary. `find` matches these against the index
# entry, which additionally pins area and shape; a global 0.25 grid resolves to exactly one
# entry per method.
GRID_IN = {"grid": [0.25, 0.25]}
GRID_OUT = {"grid": "N320"}


def resolve(method: str):
    """The index entry and the loaded scipy matrix, or a hard error explaining what is missing."""
    entry = SYS_DB.find_entry(GRID_IN, GRID_OUT, method)
    if entry is None:
        raise SystemExit(
            f"No {method} matrix for {GRID_IN} -> {GRID_OUT} in earthkit's index.\n"
            "The index lives at "
            "https://sites.ecmwf.int/repository/earthkit/regrid/db/1/index.json.gz "
            "and needs network access on first use."
        )
    return entry, SYS_DB.load_matrix(entry).tocsr()


def describe(entry, z) -> str:
    inter = entry["interpolation"]
    return (
        f"  entry       {entry['_name'][:16]}...\n"
        f"  method      {inter['engine']} v{inter['version']} {inter['method']}\n"
        f"  input       {json.dumps(entry['input'])}\n"
        f"  output      {json.dumps(entry['output'])}\n"
        f"  matrix      {z.shape[0]} x {z.shape[1]}, {z.nnz} non-zeros "
        f"({z.nnz / z.shape[0]:.2f} per target point)"
    )


def check(z) -> None:
    """Cheap invariants that would otherwise surface as a subtly wrong forecast."""
    row_sums = np.asarray(z.sum(axis=1)).ravel()
    empty = int((z.indptr[1:] == z.indptr[:-1]).sum())
    print(
        f"  row sums    [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        f"  ({empty} empty rows)"
    )
    # An interpolation is an average: every target point must be covered, and its weights
    # must sum to one. A row summing to zero would silently zero that grid point.
    assert empty == 0, f"{empty} target points have no source points"
    assert np.allclose(row_sums, 1.0, atol=1e-6), "weights do not sum to 1 per target point"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, metavar="PATH",
                        help=f"output safetensors (default: {DEFAULT_OUT.relative_to(ROOT)})")
    parser.add_argument("--method", default="linear", metavar="NAME",
                        help="interpolation method in earthkit's index (default: linear)")
    parser.add_argument("--info", action="store_true",
                        help="resolve and describe the matrix, then exit without writing")
    args = parser.parse_args()

    entry, z = resolve(args.method)
    print(f"{GRID_IN['grid']} -> {GRID_OUT['grid']}")
    print(describe(entry, z))
    check(z)
    if args.info:
        return

    # indptr indexes into indices/weights, so it is the one array that must stay wide enough
    # for nnz rather than for the grid: 1.5M fits int32 with room to spare, and int32 is what
    # Burn's Int element narrows to anyway.
    tensors = {
        "indptr": z.indptr.astype(np.int32),
        "indices": z.indices.astype(np.int32),
        "weights": z.data.astype(np.float32),
        # [target, source]. Carried as a tensor rather than safetensors metadata so the Rust
        # side reads it the same way it reads everything else.
        "shape": np.array(z.shape, dtype=np.int32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))

    size = args.out.stat().st_size / 1e6
    print(f"\nWrote {args.out} ({size:.1f} MB)")
    for name, value in tensors.items():
        print(f"  {name:8} {str(value.dtype):8} {list(value.shape)}")


if __name__ == "__main__":
    main()
