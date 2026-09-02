# /// script
# requires-python = ">=3.11"
# dependencies = ["pygrib", "numpy", "matplotlib"]
# ///
"""Plot one field from a GRIB file as a global map, PNG out.

Written for the forecast the Rust runtime writes to data/output/, but it reads any GRIB
that ecCodes can: the input open-data files and lsm.grib work too, which is how you compare
the 6 h forecast against the analysis it started from.

GRID
----
The forecast is on the N320 reduced Gaussian grid: 640 latitude rows with a different
number of points per row, no rectangle. pygrib expands that to a regular 1280 x 640 array
(expand_reduced=True, its default) by repeating each row's values, which is what
pcolormesh wants; the expansion is a display convenience and the numbers are unchanged.
The input files are 0.25 degree regular lat/lon and need no expansion.

COLOUR
------
Magnitudes take a single-direction perceptual map (viridis); a rainbow/jet map invents
boundaries where the data has none and is unreadable for colourblind readers. Signed
fields -- u/v wind, the 100u/10u/u_* family -- are drawn on a diverging map centred at 0
so that sign is the first thing the eye reads. Override either with --cmap.

Coastlines are drawn from the land-sea mask in data/grib/lsm.grib (the 0.5 contour), so
there is no cartopy dependency. --no-coast skips it.

USAGE
-----
    python scripts/plot_grib.py data/output/20260831000000-6h.grib2 --names   # what is in it
    python scripts/plot_grib.py data/output/20260831000000-6h.grib2           # 2t
    python scripts/plot_grib.py data/output/20260831000000-6h.grib2 -w shortName=z,level=500
    python scripts/plot_grib.py data/output/20260831000000-6h.grib2 -w shortName=tp --cmap Blues
    python scripts/plot_grib.py data/output/20260831000000-6h.grib2 -w shortName=msl -o msl.png
    python scripts/plot_grib.py data/output/20260831000000-6h.grib2 --temperature kelvin
    python scripts/plot_grib.py data/grib/20260831000000-0h-oper-fc.grib2 -w shortName=2t --show

`-w` takes ecCodes-style key=value pairs, comma separated; the first message matching all
of them is plotted. `-o` names the PNG; without it one is written beside the input.
`--names` lists the shortNames in the file, `--list` every message.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pygrib

ROOT = Path(__file__).resolve().parent.parent
LSM_FILE = ROOT / "data" / "grib" / "lsm.grib"

# Parameters whose sign carries meaning, so the map is centred at zero.
SIGNED = {"u", "v", "10u", "10v", "100u", "100v", "w", "vo", "d"}


def parse_where(text: str) -> dict:
    where = {}
    for pair in filter(None, text.split(",")):
        key, _, value = pair.partition("=")
        if not _:
            sys.exit(f"error: -w expects key=value pairs, got {pair!r}")
        # pygrib compares against the key's native type; level is the common integer one.
        where[key] = int(value) if value.lstrip("-").isdigit() else value
    return where


def select(grbs: pygrib.open, where: dict):
    try:
        return grbs.select(**where)[0]
    except ValueError:
        sys.exit(f"error: no message matches {where}")


def print_names(grbs: pygrib.open) -> None:
    """One line per shortName: its long name, level type, and the levels present."""
    seen: dict[str, tuple[str, str, list]] = {}
    for grb in grbs:
        name, type_of_level, levels = seen.setdefault(
            grb.shortName, (grb.name, grb.typeOfLevel, [])
        )
        levels.append(grb.level)
    width = max(len(short) for short in seen)
    for short, (name, type_of_level, levels) in sorted(seen.items()):
        if type_of_level == "isobaricInhPa":
            where = f"{type_of_level} {' '.join(str(l) for l in sorted(set(levels)))}"
        else:
            where = type_of_level
        print(f"{short:<{width}}  {name:<40} {where}")


def coastline(ax) -> None:
    if not LSM_FILE.exists():
        print(f"note: {LSM_FILE} missing, no coastline", file=sys.stderr)
        return
    with pygrib.open(str(LSM_FILE)) as grbs:
        lsm = grbs.message(1)
        values = lsm.values
        lats, lons = lsm.latlons()
    # The mask is on 0..360 and the open-data inputs on -180..180. Draw both offsets and let
    # the axis limits clip; shifting the array instead would break its column order.
    for offset in (0.0, -360.0):
        ax.contour(lons + offset, lats, values, levels=[0.5], colors="black", linewidths=0.4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=Path, help="GRIB file to read")
    parser.add_argument("-w", "--where", default="shortName=2t", help="key=value[,key=value]")
    parser.add_argument(
        "-o", "--out", type=Path, help="PNG to write (default: <input>-<where>.png next to it)"
    )
    parser.add_argument("--cmap", help="matplotlib colormap name")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--temperature",
        choices=["celsius", "kelvin"],
        default="celsius",
        help="unit for fields GRIB stores in K (default: celsius); others are unaffected",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-coast", action="store_true")
    parser.add_argument(
        "--show", action="store_true", help="open a window as well as writing the PNG"
    )
    parser.add_argument("--list", action="store_true", help="list every message and exit")
    parser.add_argument(
        "--names", action="store_true", help="list the shortNames with their levels and exit"
    )
    args = parser.parse_args()

    if not args.file.exists():
        sys.exit(f"error: {args.file} not found")

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with pygrib.open(str(args.file)) as grbs:
        if args.list:
            for grb in grbs:
                print(grb)
            return 0
        if args.names:
            print_names(grbs)
            return 0

        grb = select(grbs, parse_where(args.where))
        values = grb.values  # masked array; the bitmap becomes the mask
        lats, lons = grb.latlons()
        short_name = grb.shortName
        title = f"{grb.name} ({short_name})"
        if grb.typeOfLevel == "isobaricInhPa":
            title += f" {grb.level} hPa"
        title += f"  {grb.validDate:%Y-%m-%d %H:%M} UTC, step {grb.endStep} h"
        units = grb.units

    # GRIB keeps every temperature in kelvin; only the display changes here.
    if units == "K" and args.temperature == "celsius":
        values = values - 273.15
        units = "\N{DEGREE SIGN}C"

    finite = values.compressed() if np.ma.isMaskedArray(values) else values.ravel()
    if finite.size == 0:
        sys.exit("error: every point is missing")

    signed = short_name in SIGNED
    if signed:
        bound = max(abs(np.percentile(finite, 1)), abs(np.percentile(finite, 99)))
        vmin, vmax = -bound, bound
        cmap = args.cmap or "RdBu_r"
    else:
        vmin, vmax = np.percentile(finite, [1, 99])
        cmap = args.cmap or "viridis"
    if args.vmin is not None:
        vmin = args.vmin
    if args.vmax is not None:
        vmax = args.vmax

    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    mesh = ax.pcolormesh(lons, lats, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if not args.no_coast:
        coastline(ax)
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title, loc="left")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.colorbar(mesh, ax=ax, label=units, shrink=0.85, pad=0.02)
    ax.text(
        0.0, -0.11,
        f"min {finite.min():.4g}   max {finite.max():.4g}   mean {finite.mean():.4g}   "
        f"missing {int(np.ma.count_masked(values)):,} of {values.size:,}",
        transform=ax.transAxes, fontsize=8, color="0.35",
    )

    slug = args.where.replace("=", "_").replace(",", "-")
    out = args.out or args.file.with_name(f"{args.file.stem}-{slug}.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
