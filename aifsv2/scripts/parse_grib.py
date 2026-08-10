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

Usage:
    python scripts/parse_grib.py                       # summary + grid
    python scripts/parse_grib.py --keys                # all key names
    python scripts/parse_grib.py -q gridType 'N*' pl   # key/glob lookup
    python scripts/parse_grib.py --stats               # value statistics
    python scripts/parse_grib.py --json                # -> lsm_keys.json
    python scripts/parse_grib.py --json -              # JSON to stdout
    python scripts/parse_grib.py other.grib -m 3 -s    # 3rd message

Requires pygrib (which brings eccodes).
"""

import argparse
import json
from datetime import date, datetime
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pygrib

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "lsm.grib"

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
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_PATH, help="Path to .grib file")
    parser.add_argument("--message", "-m", type=int, default=1, metavar="N",
                        help="1-based message to inspect (default: 1)")
    parser.add_argument("--all", "-a", action="store_true", help="Apply --query/--stats to every message")
    parser.add_argument("--keys", "-k", action="store_true", help="Print all eccodes key names and exit")
    parser.add_argument("--query", "-q", nargs="+", metavar="PATTERN",
                        help="Key names or glob patterns (e.g. 'latitudeOf*'); prints matching key values")
    parser.add_argument("--stats", "-s", action="store_true", help="Print value statistics")
    parser.add_argument("--expand", action="store_true",
                        help="Stat the reduced grid expanded to a full 2-D grid (pygrib's default) "
                             "rather than the values as stored")
    parser.add_argument("--json", "-j", nargs="?", const="", metavar="OUT",
                        help="Dump every key of every message to JSON ('-' for stdout, "
                             "default: <name>_keys.json beside the input)")
    parser.add_argument("--max-array", type=int, default=1024, metavar="N",
                        help="Arrays longer than N are summarised in JSON instead of inlined (default: 1024)")
    args = parser.parse_args()

    grbs = pygrib.open(str(args.path))
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

        if args.json is not None:
            grbs.rewind()
            dump = {"path": str(args.path), "messages": []}
            for msg in grbs:
                entry = {"messagenumber": msg.messagenumber}
                entry |= {k: jsonable(safe_get(msg, k), args.max_array) for k in key_names(msg)}
                dump["messages"].append(entry)

            text = json.dumps(dump, indent=2, default=str)
            if args.json == "-":
                print(text)
            else:
                out = Path(args.json) if args.json else args.path.with_name(args.path.stem + "_keys.json")
                out.write_text(text)
                print(f"Wrote {len(dump['messages'])} message(s) to {out}")
            return

        # Default: file summary, per-message listing, then the grid of the first message.
        print(f"{args.path}")
        plural = "message" if grbs.messages == 1 else "messages"
        print(f"  {args.path.stat().st_size / 1e6:.1f} MB, {grbs.messages} {plural}, "
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
