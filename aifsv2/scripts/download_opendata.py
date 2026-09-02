"""Download the ECMWF open-data `oper` and `wave` GRIB inputs for AIFS-single, on demand.

WHAT THIS FETCHES
-----------------
`inference.yaml` sets `input: opendata`: the model is initialised from ECMWF's
real-time open-data feed, which ships two files per forecast base time —

  {YYYYMMDDHHMMSS}-{step}h-oper-fc.grib2   operational atmosphere (pl + sfc)
  {YYYYMMDDHHMMSS}-{step}h-wave-fc.grib2   wave model (sfc)

matching the sample files documented in docs/grib-input-explained.md §2.1–2.2.
This script reproduces those files for any base time (or a range of them) using
the same `ecmwf-opendata` client that wrote the `.index` sidecars on disk.

WHERE THE FIELD LIST COMES FROM
-------------------------------
The exact params/levels are read from the checkpoint's own metadata
(`aifs-single-mse-2.0_metadata.json`, the dataset.variables_metadata block), so
the download always tracks what the model consumes. Two adjustments the metadata
forces:

  * Computed forcings (insolation, cos_latitude, …) carry no `mars` block — they
    are generated at inference time, never downloaded, so they fall out naturally.
  * `cos_mwd` / `sin_mwd` are produced by the `cos_sin_mean_wave_direction`
    pre-processor from `mwd`; open data only has `mwd`. They are excluded here,
    which brings the wave request to the 11 real messages (§2.2).
  * The soil fields are archived as `stl{1,2}`/`swvl{1,2}` (levtype sfc) but
    published in open data as `sot`/`vsw` levels 1/2 (levtype sol); they are
    translated here and renamed back on read (src/grib.rs SOIL_RENAMES).

Params are grouped into one retrieval per (stream, levtype, level-set): pressure
levels split into {t,u,v,z,w} @ 14 levels and {q} @ 13 levels (no 10 hPa). Each
group is fetched to a temp part and the parts for a stream are concatenated
(GRIB messages append byte-for-byte) into the final file.

Note: `lsm.grib` is NOT fetched here. It is an N320 native-resolution MARS field,
not part of the open-data retrieval (docs §2.3); it ships with the repo.

RETENTION
---------
ECMWF only keeps roughly the last ~4 days of open data online; older base times
are purged (and future ones do not exist yet). Absolute dates outside that window
are warned about before they 404. Use `--latest` to grab the newest published
run, or relative day offsets (`--start -1 --end 0`) to stay inside the window.

USAGE
-----
    # newest available run, both streams, step 0
    uv run python scripts/download_opendata.py --latest --output-dir data

    # the last two days at 00Z and 12Z, into data/
    uv run python scripts/download_opendata.py \
        --start -1 --end 0 --freq 12 --output-dir data

    # a specific base time
    uv run python scripts/download_opendata.py --start 2026-09-02T00 --end 2026-09-02T00

    # see exactly what would be requested without downloading
    uv run python scripts/download_opendata.py --latest --dry-run
"""

import argparse
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Open-data forecast cycles. Base times outside this set have no product.
VALID_CYCLES = {0, 6, 12, 18}

# ECMWF only keeps roughly the last few days of open data online (older base
# times are purged). Requests older than this warn rather than silently 404.
DEFAULT_MAX_AGE_DAYS = 4

# Model variables that exist in the checkpoint but are never in open data:
# derived by a pre-processor from a param that IS downloaded.
DERIVED_PARAMS = {"cos_mwd", "sin_mwd"}

# The checkpoint metadata names soil fields the way the MARS training archive
# does: stl{1,2}/swvl{1,2}, levtype sfc, no levelist. Open data publishes the
# same fields as sot/vsw with levelist 1/2 under levtype `sol`, so requesting
# the archive names returns nothing. This is the download-side half of the
# rename; src/grib.rs SOIL_RENAMES undoes it on read.
#   archive param -> (open-data param, levtype, levelist)
MARS_TO_OPENDATA = {
    "stl1": ("sot", "sol", 1),
    "stl2": ("sot", "sol", 2),
    "swvl1": ("vsw", "sol", 1),
    "swvl2": ("vsw", "sol", 2),
}

DEFAULT_METADATA = "aifs-single-mse-2.0_metadata.json"


def parse_base_time(text: str) -> datetime:
    """Accept an absolute date/time or a relative day offset (all UTC).

    Absolute:  YYYY-MM-DD (00Z), YYYY-MM-DDTHH, YYYY-MM-DDTHH:MM, or YYYYMMDD.
    Relative:  a signed integer number of days from today 00Z
               (0 = today, -1 = yesterday). Convenient because open data only
               retains roughly the last few days.
    """
    text = text.strip().replace(" ", "T")
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
    if re.fullmatch(r"[+-]?\d{1,3}", text):
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return today + timedelta(days=int(text))
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"unrecognised date/time {text!r}; use YYYY-MM-DD, YYYY-MM-DDTHH, "
        "YYYYMMDD, or a signed day offset like -1"
    )


def base_times(start: datetime, end: datetime, freq_hours: int):
    if end < start:
        raise SystemExit("--end is before --start")
    if freq_hours <= 0:
        raise SystemExit("--freq must be a positive number of hours")
    t = start
    while t <= end:
        yield t
        t += timedelta(hours=freq_hours)


def load_metadata_variables(metadata_path: Path) -> dict:
    """Return the dataset.variables_metadata mapping from an extracted
    <ckpt>_metadata.json (produced by ckpt_to_safetensors.py)."""
    import json

    doc = json.loads(metadata_path.read_text())
    # The file is keyed by the json path inside the ckpt zip; there is one entry.
    for content in doc.values():
        if isinstance(content, dict) and "dataset" in content:
            return content["dataset"]["variables_metadata"]
    raise SystemExit(
        f"{metadata_path}: no dataset.variables_metadata block found "
        "(is this a *_metadata.json from ckpt_to_safetensors.py?)"
    )


def build_groups(variables_metadata: dict, streams: set) -> dict:
    """(stream, levtype, frozenset(levels)) -> sorted list of params.

    Groups params that share a stream/levtype/level-set into a single retrieval.
    """
    per_param_levels = defaultdict(set)
    for meta in variables_metadata.values():
        mars = meta.get("mars")
        if not mars:
            continue
        param = mars.get("param")
        stream = mars.get("stream")
        if not param or not stream or stream not in streams:
            continue
        if param in DERIVED_PARAMS:
            continue
        levtype = mars.get("levtype", "sfc")
        level = mars.get("levelist")
        if param in MARS_TO_OPENDATA:
            param, levtype, level = MARS_TO_OPENDATA[param]
        per_param_levels[(stream, levtype, param)].add(
            int(level) if level is not None else None
        )

    groups = defaultdict(set)
    for (stream, levtype, param), levels in per_param_levels.items():
        level_set = frozenset(lv for lv in levels if lv is not None)
        groups[(stream, levtype, level_set)].add(param)

    # Deterministic order: oper before wave, pl before sfc, then by level count.
    def sort_key(key):
        stream, levtype, level_set = key
        return (stream != "oper", levtype != "pl", -len(level_set))

    return {k: sorted(groups[k]) for k in sorted(groups, key=sort_key)}


def describe_groups(groups: dict) -> None:
    for (stream, levtype, level_set), params in groups.items():
        levels = f" levels={sorted(level_set)}" if level_set else ""
        print(f"  [{stream}/{levtype}] {len(params)} params{levels}")
        print(f"      {','.join(params)}")


def concat(parts: list, target: Path) -> None:
    with target.open("wb") as out:
        for part in parts:
            out.write(Path(part).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=parse_base_time,
                        help="First forecast base time: YYYY-MM-DD, YYYY-MM-DDTHH, "
                             "YYYYMMDD, or a signed day offset like -3 (UTC).")
    parser.add_argument("--end", type=parse_base_time,
                        help="Last forecast base time, inclusive (same formats as --start).")
    parser.add_argument("--latest", action="store_true",
                        help="Fetch the single most recent base time ECMWF has published "
                             "(ignores --start/--end/--freq).")
    parser.add_argument("--freq", type=int, default=24,
                        help="Hours between base times in the range (default 24).")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                        help="Warn if a base time is older than this; open data retains "
                             f"~{DEFAULT_MAX_AGE_DAYS} days (default: {DEFAULT_MAX_AGE_DAYS}).")
    parser.add_argument("--step", type=int, nargs="+", default=[0],
                        help="Forecast step hour(s) to fetch (default: 0).")
    parser.add_argument("--stream", nargs="+", default=["oper", "wave"],
                        choices=["oper", "wave"],
                        help="Which streams to download (default: both).")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for the .grib2 files (default: current dir).")
    parser.add_argument("--metadata", type=Path, default=None,
                        help=f"Path to <ckpt>_metadata.json (default: ./{DEFAULT_METADATA} "
                             "or next to --output-dir).")
    parser.add_argument("--model", default="aifs-single", help="Open-data model (default: aifs-single).")
    parser.add_argument("--resol", default="0p25", help="Open-data resolution (default: 0p25).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if the target file already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned retrievals and exit without downloading.")
    args = parser.parse_args()

    if not args.latest and (args.start is None or args.end is None):
        parser.error("provide both --start and --end, or use --latest")

    streams = set(args.stream)

    # Locate metadata: explicit, else cwd, else alongside the output dir.
    candidates = [args.metadata] if args.metadata else [
        Path(DEFAULT_METADATA),
        args.output_dir / DEFAULT_METADATA,
    ]
    metadata_path = next((p for p in candidates if p and p.is_file()), None)
    if metadata_path is None:
        raise SystemExit(
            "Could not find the checkpoint metadata JSON. Pass --metadata "
            f"<path to *_metadata.json>. Looked in: {[str(p) for p in candidates if p]}"
        )

    variables_metadata = load_metadata_variables(metadata_path)
    groups = build_groups(variables_metadata, streams)
    if not groups:
        raise SystemExit(f"No downloadable params for streams {sorted(streams)}")

    print(f"Field spec from {metadata_path}:")
    describe_groups(groups)

    now = datetime.now(timezone.utc)

    def warn_base_time(bt: datetime) -> None:
        if bt.hour not in VALID_CYCLES:
            print(f"  warning: {bt:%Y-%m-%d %HZ} is not an open-data cycle "
                  f"{sorted(VALID_CYCLES)} — retrieval will likely fail.", file=sys.stderr)
        if bt > now + timedelta(hours=1):
            print(f"  warning: {bt:%Y-%m-%d %HZ} is in the future — not yet produced.",
                  file=sys.stderr)
        else:
            age_days = (now - bt).total_seconds() / 86400
            if age_days > args.max_age_days:
                print(f"  warning: {bt:%Y-%m-%d %HZ} is ~{age_days:.1f} days old; open data "
                      f"keeps ~{args.max_age_days} days, so this will likely 404.",
                      file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --latest needs the network to resolve; an explicit range is known up front.
    if args.dry_run:
        if args.latest:
            print("\nWould fetch the latest available base time "
                  "(resolved from ECMWF at download time):")
            for stream in args.stream:
                n_groups = sum(1 for (s, _, _) in groups if s == stream)
                print(f"  would write <latest>-<step>h-{stream}-fc.grib2  ({n_groups} request(s))")
        else:
            plan = [(bt, step) for bt in base_times(args.start, args.end, args.freq)
                    for step in args.step]
            print(f"\n{len(plan)} base-time/step combination(s) x {len(streams)} stream(s):")
            for bt, step in plan:
                warn_base_time(bt)
                stamp = bt.strftime("%Y%m%d%H%M%S")
                for stream in args.stream:
                    target = args.output_dir / f"{stamp}-{step}h-{stream}-fc.grib2"
                    n_groups = sum(1 for (s, _, _) in groups if s == stream)
                    print(f"  would write {target}  ({n_groups} request(s))")
        print("\n(dry run — nothing downloaded)")
        return 0

    # Import here so --dry-run and --help work without the client installed.
    from ecmwf.opendata import Client

    client = Client(source="ecmwf", model=args.model, resol=args.resol)

    if args.latest:
        latest_bt = client.latest({"stream": args.stream[0], "type": "fc"})
        print(f"\nLatest available base time: {latest_bt:%Y-%m-%d %HZ}")
        base_list = [latest_bt]
    else:
        base_list = list(base_times(args.start, args.end, args.freq))

    plan = [(bt, step) for bt in base_list for step in args.step]
    print(f"{len(plan)} base-time/step combination(s) x {len(streams)} stream(s):")
    for bt, _ in plan:
        warn_base_time(bt)

    failures = 0
    for bt, step in plan:
        stamp = bt.strftime("%Y%m%d%H%M%S")
        date = int(bt.strftime("%Y%m%d"))
        time = bt.hour
        for stream in args.stream:
            target = args.output_dir / f"{stamp}-{step}h-{stream}-fc.grib2"
            if target.exists() and not args.overwrite:
                print(f"  skip (exists): {target}")
                continue

            stream_groups = {k: v for k, v in groups.items() if k[0] == stream}
            parts = []
            try:
                with tempfile.TemporaryDirectory(dir=args.output_dir) as tmp:
                    for i, ((_, levtype, level_set), params) in enumerate(stream_groups.items()):
                        request = {
                            "date": date, "time": time, "step": step,
                            "stream": stream, "type": "fc",
                            "levtype": levtype, "param": params,
                        }
                        if level_set:
                            request["levelist"] = sorted(level_set)
                        part = Path(tmp) / f"part{i}.grib2"
                        print(f"  {target.name}: retrieving {stream}/{levtype} "
                              f"({len(params)} params)...")
                        client.retrieve(request, target=str(part))
                        parts.append(part)
                    concat(parts, target)
                size_mb = target.stat().st_size / 1e6
                print(f"  wrote {target} ({size_mb:.1f} MB)")
            except Exception as exc:  # noqa: BLE001 - report and continue the batch
                failures += 1
                if target.exists():
                    target.unlink()
                print(f"  FAILED {target.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"\nDone with {failures} failure(s).", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
