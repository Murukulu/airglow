# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Extract anemoi-inference's built-in N320 GRIB templates to data/templates/.

WHY
---
Output GRIB is written by cloning a template message and overwriting its keys and values;
neither ecCodes nor the Rust crate can build a message from nothing. The templates have to
be on the model grid, and nothing in data/grib/ is: the open-data forecasts are 0.25 degree
regular lat/lon, and lsm.grib is N320 but GRIB edition 1 with simple packing, a poor base
for a forecast product.

anemoi-inference solves the same problem with a small set of zeroed GRIB2/CCSDS messages,
one per (grid, levtype), stored zlib+base64 in
src/anemoi/inference/grib/templates/builtin.yaml and selected by the `builtin` template
provider (grib/templates/builtin.py). This script pulls the two N320 entries out of that
index -- `{grid: N320, levtype: pl}` and `{grid: N320}` (surface) -- and writes them as plain
GRIB files, 1,480 bytes each, that src/output.rs reads with CodesFile::new_from_memory.

The decoded files are committed, so the Rust build does not depend on the anemoi checkout.
Re-run only to pick up an upstream change to the templates.

USAGE
-----
    uv run scripts/extract_grib_templates.py [--anemoi ../anemoi-inference] [--out data/templates]
"""

import argparse
import base64
import sys
import zlib
from pathlib import Path

import yaml

BUILTIN_INDEX = Path("src/anemoi/inference/grib/templates/builtin.yaml")

# (lookup that must match the index entry exactly, output filename)
WANTED = [
    ({"grid": "N320", "levtype": "pl"}, "n320-pl.grib2"),
    ({"grid": "N320"}, "n320-sfc.grib2"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--anemoi", type=Path, default=Path("../anemoi-inference"),
                        help="anemoi-inference checkout (default: ../anemoi-inference)")
    parser.add_argument("--out", type=Path, default=Path("data/templates"),
                        help="output directory (default: data/templates)")
    args = parser.parse_args()

    index_path = args.anemoi / BUILTIN_INDEX
    if not index_path.is_file():
        print(f"error: {index_path} not found; pass --anemoi", file=sys.stderr)
        return 1

    with index_path.open() as f:
        entries = yaml.safe_load(f)

    args.out.mkdir(parents=True, exist_ok=True)
    for lookup, filename in WANTED:
        matches = [blob for entry_lookup, blob in entries if entry_lookup == lookup]
        if len(matches) != 1:
            print(
                f"error: expected exactly one entry for {lookup}, found {len(matches)}",
                file=sys.stderr,
            )
            return 1

        raw = zlib.decompress(base64.b64decode(matches[0]))
        if raw[:4] != b"GRIB":
            print(f"error: decoded entry for {lookup} is not GRIB", file=sys.stderr)
            return 1

        dest = args.out / filename
        dest.write_bytes(raw)
        print(f"wrote {dest} ({len(raw)} bytes, edition {raw[7]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
