"""Inspect a GRIB file: summarise messages, query eccodes keys, dump metadata to JSON.

Written for data/lsm.grib — the land-sea mask that data/inference.yaml feeds to the
apply-mask pre-processor for sd/swvl1/swvl2. That file is a single GRIB1 message on the
N320 reduced Gaussian grid (542,080 points), which is the grid AIFS itself runs on, so
this doubles as a way to read out the grid layout for the Burn port.

REDUCED GAUSSIAN GRIDS
----------------------
There is no Ni x Nj rectangle. The globe is split into 2N latitude rows (N=320 -> 640
rows) and each row gets its own point count, listed in the `pl` key. Values are stored as
one flat array in row-major order, so reproducing the grid means keeping `pl` intact —
hence --max-array defaults to 1024, big enough to inline all 640 entries of `pl` while
still collapsing the 542,080-element `values`/`latitudes`/`longitudes` arrays.

MISSING VALUES
--------------
Several fields are undefined over part of the globe: every wave field is defined only over
water, and soil/snow fields only over land. GRIB expresses that with a *bitmap* — a per-point
validity mask in section 6 — and ecCodes expands it for you, filling the masked points with
`missingValue` (9999 unless something sets it otherwise). src/grib.rs:248 rewrites those to
NaN at the boundary, because the checkpoint's ConstantImputer only recognises NaN and 9999
would otherwise reach the model as a real measurement.

--missing reports that accounting per message. --regrid additionally pushes the field through
the same 0.25 -> N320 operator the Rust runtime uses, which is how you check whether a NaN
count on the model grid is source data or something the port introduced: NaN propagates
through the interpolation stencil, so any target point whose stencil touches a masked source
point comes out NaN too.

Usage:
    python scripts/parse_grib.py                       # summary + grid
    python scripts/parse_grib.py --keys                # all key names
    python scripts/parse_grib.py -q gridType 'N*' pl   # key/glob lookup
    python scripts/parse_grib.py --stats               # value statistics
    python scripts/parse_grib.py --json                # -> lsm_keys.json
    python scripts/parse_grib.py --json -              # JSON to stdout
    python scripts/parse_grib.py -f other.grib -m 3 -s  # 3rd message of another file
    python scripts/parse_grib.py -f data/20260810000000-0h-wave-fc.grib2 -a --missing
    python scripts/parse_grib.py -f data/20260810000000-0h-wave-fc.grib2 -a --regrid
    ... --regrid --transform cos -q shortName          # cos(mwd), as the port builds it

Requires pygrib (which brings eccodes). --regrid additionally needs scipy and safetensors;
all three are in data/aifs-single-mse-2.0/quiet_grub/.venv.
"""

import argparse
import json
from datetime import date, datetime
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pygrib

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "lsm.grib"
DEFAULT_MATRIX = ROOT / "data" / "regrid-0p25-to-n320.safetensors"

# Keys that are huge, redundant with `values`, or that eccodes only exposes as raw
# section bytes. Skipped when enumerating keys so a dump stays readable.
SKIP_KEYS = {"values", "codedValues", "latitudes", "longitudes", "distinctLatitudes",
             "distinctLongitudes", "latLonValues", "section8", "7777"}


def matches(key: str, pattern: str) -> bool:
    return fnmatch(key, pattern) if any(c in pattern for c in "*?[") else pattern in key


def key_names(msg) -> list[str]:
    """Sorted eccodes key names for a message, minus the bulk-data keys."""
    return sorted(k for k in msg.keys() if k not in SKIP_KEYS)


def safe_get(msg, key):
    """Read one key, tolerating the ones eccodes refuses.

    A few names pygrib lists (validDate, analDate) are computed properties rather than
    real eccodes keys, so the item lookup raises and the attribute lookup works.
    """
    try:
        return msg[key]
    except Exception as e:
        try:
            return getattr(msg, key)
        except Exception:
            return f"<unreadable: {e}>"


def fmt_value(value) -> str:
    """Render one key value on a single line — `pl` alone is 640 entries."""
    if isinstance(value, np.ndarray):
        head = ", ".join(f"{v:g}" for v in np.ravel(value)[:8])
        return (f"{value.dtype}[{value.size}] min={value.min():g} max={value.max():g} "
                f"sum={value.sum():g} head=[{head}, ...]")
    return str(value)


def jsonable(value, max_array: int):
    """Coerce an eccodes value into something json.dumps can write."""
    if isinstance(value, np.ndarray):
        data = value.compressed() if np.ma.isMaskedArray(value) else value
        if data.size <= max_array:
            return data.tolist()
        return {"__array__": {"len": int(data.size), "dtype": str(data.dtype),
                              "min": data.min().item(), "max": data.max().item(),
                              "head": data[:8].tolist()}}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def summarise_message(msg) -> str:
    """One-line description of a message, for the default summary listing."""
    level = f"{safe_get(msg, 'typeOfLevel')}={safe_get(msg, 'level')}"
    return (f"  #{msg.messagenumber:<3d} {str(safe_get(msg, 'shortName')):10s} "
            f"{str(safe_get(msg, 'name')):32s} [{safe_get(msg, 'units')}] {level} "
            f"{msg.validDate} {safe_get(msg, 'gridType')} "
            f"{safe_get(msg, 'numberOfDataPoints')} pts")


def describe_grid(msg) -> dict:
    """Grid geometry and packing — enough to rebuild the point layout elsewhere."""
    info = {k: safe_get(msg, k) for k in
            ("gridType", "numberOfDataPoints", "numberOfValues", "missingValue",
             "packingType", "bitsPerValue", "latitudeOfFirstGridPointInDegrees",
             "longitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
             "longitudeOfLastGridPointInDegrees")}

    if msg.has_key("N"):
        info["N"] = safe_get(msg, "N")
    # Ni is set to the eccodes MISSING sentinel on reduced grids — rows differ in width.
    if msg.has_key("Ni") and msg.has_key("Nj") and not msg.is_missing("Ni"):
        info["Ni x Nj"] = f"{safe_get(msg, 'Ni')} x {safe_get(msg, 'Nj')}"
    if msg.has_key("pl"):
        pl = safe_get(msg, "pl")
        if isinstance(pl, np.ndarray):
            info["pl (points per row)"] = (f"{pl.size} rows, {pl.min()}..{pl.max()} per row, "
                                           f"{int(pl.sum())} total")
    return info


def value_stats(msg, expand: bool = False, max_unique: int = 20) -> dict:
    """Distribution of the message's data values.

    pygrib expands reduced grids to a full 2-D lat/lon grid by default, which for this
    file turns the 542,080 stored values into 640x1280 = 819,200 interpolated ones. We
    turn that off so the statistics describe what is actually in the file; --expand asks
    for pygrib's default view instead.

    Fields with few distinct values are far better described by a value->count table than
    by percentiles, so we pick per message.
    """
    msg.expand_grid(expand)
    values = msg.values
    data = values.compressed() if np.ma.isMaskedArray(values) else np.asarray(values)
    finite = data[np.isfinite(data)]

    stats = {"points": int(np.size(values)), "missing": int(np.size(values) - finite.size)}
    if finite.size == 0:
        return stats

    stats |= {"min": float(finite.min()), "max": float(finite.max()),
              "mean": float(finite.mean()), "std": float(finite.std())}

    uniq, counts = np.unique(finite, return_counts=True)
    if uniq.size <= max_unique:
        stats["values"] = {f"{v:g}": f"{c} ({100 * c / finite.size:.2f}%)"
                           for v, c in zip(uniq, counts)}
    else:
        stats["distinct"] = int(uniq.size)
        pcts = np.percentile(finite, [1, 25, 50, 75, 99])
        stats |= {f"p{p}": float(v) for p, v in zip((1, 25, 50, 75, 99), pcts)}

    # Where the extremes sit. latlons() is not supported for every grid type.
    try:
        lats, lons = msg.latlons()
        flat = np.ravel(values)
        for label, idx in (("argmin", int(np.nanargmin(flat))), ("argmax", int(np.nanargmax(flat)))):
            stats[label] = f"index {idx} at lat {np.ravel(lats)[idx]:.3f}, lon {np.ravel(lons)[idx]:.3f}"
    except Exception as e:
        stats["argmin/argmax"] = f"<no lat/lon: {e}>"
    return stats


def stored_values(msg, expand: bool = False) -> np.ndarray:
    """The message's values as a flat float64 array, masked points as NaN.

    pygrib expands reduced grids to a full 2-D lat/lon grid unless told not to — for lsm.grib
    that would turn 542,080 stored values into 640x1280 interpolated ones — so that is turned
    off and what comes back is what is in the file, in stored order.

    Turning it off also drops pygrib's masked-array wrapper, leaving the raw `missingValue`
    fill (9999) in the data. That is precisely the array src/grib.rs receives from ecCodes,
    so the substitution below is the same one `unmask` does at src/grib.rs:248-252, and for
    the same reason: 9999 is a plausible-looking wave height, and nothing downstream would
    catch it.
    """
    msg.expand_grid(expand)
    values = msg.values
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values.astype("float64"), np.nan)
    values = np.asarray(values, dtype="float64").ravel()

    missing = safe_get(msg, "missingValue")
    if isinstance(missing, (int, float)):
        values = np.where(values == missing, np.nan, values)
    return values


def missing_report(msg, expand: bool = False) -> dict:
    """Bitmap accounting for one message: how much is masked, and what survives."""
    values = stored_values(msg, expand)
    nan = np.isnan(values)
    finite = values[~nan]

    info = {
        "shortName": safe_get(msg, "shortName"),
        "paramId": safe_get(msg, "paramId"),
        "typeOfLevel": safe_get(msg, "typeOfLevel"),
        "gridType": safe_get(msg, "gridType"),
        "points": values.size,
        "bitmapPresent": safe_get(msg, "bitmapPresent") if msg.has_key("bitmapPresent") else "<absent>",
        "missingValue": safe_get(msg, "missingValue"),
        "missing": f"{int(nan.sum())} ({100 * nan.mean():.1f}%)",
        "finite": int(finite.size),
    }
    if finite.size:
        info["range"] = f"[{finite.min():.4f}, {finite.max():.4f}]"
        uniq = np.unique(finite)
        info["distinct"] = int(uniq.size)
    return info


def load_matrix(path: Path):
    """The 0.25 -> N320 CSR operator written by fetch_regrid_matrix.py."""
    from safetensors.numpy import load_file
    from scipy.sparse import csr_matrix

    t = load_file(str(path))
    num_target, num_source = (int(v) for v in t["shape"])
    matrix = csr_matrix((t["weights"], t["indices"], t["indptr"]),
                        shape=(num_target, num_source))
    return matrix


def regrid_report(msg, matrix, transform: str | None = None) -> dict:
    """Push one message through the regrid and report what the model grid ends up holding.

    Mirrors src/grib.rs exactly: the cos/sin transform (for mwd) runs on the source grid
    before the regrid, and the row rotation runs inside it. The matrix's input gridspec has
    column 0 at Greenwich while the open-data files start at the dateline, so each row is
    rolled by longitudeOfFirstGridPointInDegrees / iDirectionIncrementInDegrees columns
    first — 720 for a 0.25 degree file. Dropping that rotation leaves the global mean intact
    and moves a third of the points, which is why it is checked here rather than assumed.
    """
    ni, nj = safe_get(msg, "Ni"), safe_get(msg, "Nj")
    lon_first = safe_get(msg, "longitudeOfFirstGridPointInDegrees")
    di = safe_get(msg, "iDirectionIncrementInDegrees")
    values = stored_values(msg)

    if values.size != matrix.shape[1] or ni * nj != matrix.shape[1]:
        return {"error": f"matrix takes {matrix.shape[1]} source points, "
                         f"field has {values.size} ({ni} x {nj})"}

    if transform:
        radians = np.radians(values)
        values = np.cos(radians) if transform == "cos" else np.sin(radians)

    shift = int(round(lon_first / di)) % ni
    rolled = values.reshape(nj, ni)
    if shift:
        rolled = np.concatenate([rolled[:, ni - shift:], rolled[:, :ni - shift]], axis=1)
    out = matrix @ rolled.ravel()

    src_nan, out_nan = np.isnan(values), np.isnan(out)
    finite = out[~out_nan]
    info = {
        "transform": transform or "none",
        "row shift": f"{shift} columns ({lon_first} deg first, {di} deg step)",
        "source": f"{values.size} points, {int(src_nan.sum())} NaN ({100 * src_nan.mean():.1f}%)",
        "target": f"{out.size} points, {int(out_nan.sum())} NaN ({100 * out_nan.mean():.1f}%)",
    }
    info["finite"] = (f"{finite.size}, range [{finite.min():.4f}, {finite.max():.4f}]"
                      if finite.size else "0 — the whole field is NaN on the model grid")
    return info


def print_block(title: str, info: dict) -> None:
    print(title)
    width = max((len(k) for k in info), default=0)
    for k, v in info.items():
        print(f"  {k:{width}s}  {v}")


def select(grbs, message: int, want_all: bool) -> list:
    if want_all:
        grbs.rewind()
        return list(grbs)
    if not 1 <= message <= grbs.messages:
        raise ValueError(f"Message {message} out of range (file has {grbs.messages})")
    return [grbs.message(message)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", "-f", type=Path, default=DEFAULT_PATH, metavar="PATH",
                        help=f"GRIB file to inspect (default: {DEFAULT_PATH.name})")
    parser.add_argument("--message", "-m", type=int, default=1, metavar="N",
                        help="1-based message to inspect (default: 1)")
    parser.add_argument("--all", "-a", action="store_true", help="Apply --query/--stats to every message")
    parser.add_argument("--keys", "-k", action="store_true", help="Print all eccodes key names and exit")
    parser.add_argument("--query", "-q", nargs="+", metavar="PATTERN",
                        help="Key names or glob patterns (e.g. 'latitudeOf*'); prints matching key values")
    parser.add_argument("--stats", "-s", action="store_true", help="Print value statistics")
    parser.add_argument("--missing", "-M", action="store_true",
                        help="Print bitmap / missing-value accounting (what src/grib.rs turns into NaN)")
    parser.add_argument("--regrid", "-r", action="store_true",
                        help="Push the field through the 0.25 -> N320 operator and report NaN either side")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, metavar="PATH",
                        help=f"CSR operator for --regrid (default: {DEFAULT_MATRIX.name})")
    parser.add_argument("--transform", choices=("cos", "sin"), metavar="FN",
                        help="Apply cos/sin of the field in degrees before regridding, as the port "
                             "does for mwd -> cos_mwd/sin_mwd")
    parser.add_argument("--expand", action="store_true",
                        help="Stat the reduced grid expanded to a full 2-D grid (pygrib's default) "
                             "rather than the values as stored")
    parser.add_argument("--json", "-j", nargs="?", const="", metavar="OUT",
                        help="Dump every key of every message to JSON ('-' for stdout, "
                             "default: <name>_keys.json beside the input)")
    parser.add_argument("--max-array", type=int, default=1024, metavar="N",
                        help="Arrays longer than N are summarised in JSON instead of inlined (default: 1024)")
    args = parser.parse_args()

    grbs = pygrib.open(str(args.file))
    try:
        if args.keys:
            for msg in select(grbs, args.message, args.all):
                print(f"Message {msg.messagenumber}: {len(key_names(msg))} keys")
                for k in key_names(msg):
                    print(f"  {k}")
            return

        if args.query:
            for msg in select(grbs, args.message, args.all):
                print(f"Message {msg.messagenumber}:")
                names = key_names(msg)
                for pattern in args.query:
                    found = [k for k in names if matches(k, pattern)]
                    if not found:
                        print(f"  (no keys matched {pattern!r})")
                        continue
                    for k in found:
                        print(f"  {k:40s} {fmt_value(safe_get(msg, k))}")
            return

        if args.stats:
            for msg in select(grbs, args.message, args.all):
                grid = "expanded 2-D grid" if args.expand else "stored values"
                print_block(f"Message {msg.messagenumber} ({safe_get(msg, 'shortName')}), {grid}:",
                            value_stats(msg, args.expand))
            return

        if args.missing:
            for msg in select(grbs, args.message, args.all):
                print_block(f"Message {msg.messagenumber}:", missing_report(msg, args.expand))
            return

        if args.regrid:
            matrix = load_matrix(args.matrix)
            print(f"{args.matrix.name}: {matrix.shape[0]} target x {matrix.shape[1]} source, "
                  f"{matrix.nnz} non-zeros, {matrix.nnz / matrix.shape[0]:.2f} per target point\n")
            for msg in select(grbs, args.message, args.all):
                print_block(f"Message {msg.messagenumber} ({safe_get(msg, 'shortName')}):",
                            regrid_report(msg, matrix, args.transform))
            return

        if args.json is not None:
            grbs.rewind()
            dump = {"path": str(args.file), "messages": []}
            for msg in grbs:
                entry = {"messagenumber": msg.messagenumber}
                entry |= {k: jsonable(safe_get(msg, k), args.max_array) for k in key_names(msg)}
                dump["messages"].append(entry)

            text = json.dumps(dump, indent=2, default=str)
            if args.json == "-":
                print(text)
            else:
                out = Path(args.json) if args.json else args.file.with_name(args.file.stem + "_keys.json")
                out.write_text(text)
                print(f"Wrote {len(dump['messages'])} message(s) to {out}")
            return

        # Default: file summary, per-message listing, then the grid of the first message.
        print(f"{args.file}")
        plural = "message" if grbs.messages == 1 else "messages"
        print(f"  {args.file.stat().st_size / 1e6:.1f} MB, {grbs.messages} {plural}, "
              f"GRIB edition {safe_get(grbs.message(1), 'editionNumber')}")

        grbs.rewind()
        for msg in grbs:
            if msg.messagenumber > 10:
                print(f"  ... and {grbs.messages - 10} more")
                break
            print(summarise_message(msg))

        print()
        first = grbs.message(1)
        print_block(f"Grid (message 1, {len(key_names(first))} keys):", describe_grid(first))
        print("\nUse --keys / --query / --stats / --json for more.")
    finally:
        grbs.close()


if __name__ == "__main__":
    main()
