# /// script
# requires-python = ">=3.11"
# dependencies = ["pygrib", "numpy", "scipy", "safetensors", "matplotlib"]
# ///
"""Score a forecast GRIB from the Rust runtime against an ECMWF analysis.

WHY
---
A forward pass that runs is not a forward pass that is right. The only end-to-end check is
to compare what the model produced against what actually happened: the 6 h forecast valid
at 06Z against the 06Z analysis. The bar is persistence -- "tomorrow is like today", here
the 00Z analysis the forecast started from. A working model beats persistence on most
fields at 6 h; a broken port (wrong channel order, missing normaliser, transposed weight)
does not, and usually loses to it by a wide margin.

WHAT IT DOES
------------
For every message in the forecast it finds the same variable in the truth file (and in
the baseline file if given), regrids those from 0.25 degree to N320 with the same operator
and row rotation src/grib.rs uses (parse_grib.regrid_values), and over the points where
both are finite reports:

  rmse     root mean square of forecast - truth
  bias     mean of forecast - truth
  corr     Pearson correlation between forecast and truth
  persist  rmse of baseline - truth, when --baseline is given
  skill    1 - rmse / persist: positive means the model beat persistence

Mean wave direction is scored on the wrapped difference (-180, 180]. The six accumulated
fields (cp ro sf ssrd strd tp) are scored only when the truth message covers an interval:
against an analysis (step 0) they are listed but skipped. Fields the open-data files do not
carry (sd) are skipped. Open-data cloud covers arrive in percent and are scaled to fractions.

The truth's wave file is found by replacing "oper" with "wave" in the truth path, so pass
the oper file. The forecast is 0..360 and the analyses -180..180; the regrid handles that.

The truth may also be an anemoi-inference output on N320 (the reference implementation run
on the same date), in which case it is used as stored. That is the comparison that tells a
port bug from a model limitation: the Rust output should agree with anemoi's to within
float noise, whatever either of them scores against the analysis.

USAGE
-----
    # Real scoring: truth is the analysis valid at the forecast time, baseline its start
    python scripts/compare_grib.py data/output/20260831000000-6h.grib2 \
        --truth data/grib/20260831060000-0h-oper-fc.grib2 \
        --baseline data/grib/20260831000000-0h-oper-fc.grib2

    # How far did 6 h move the state? Compare against the initial condition itself
    python scripts/compare_grib.py data/output/20260831000000-6h.grib2 \
        --truth data/grib/20260831000000-0h-oper-fc.grib2

    # Port check: the Rust output against anemoi-inference's own output for the same date
    python scripts/compare_grib.py data/output/20260831000000-6h.grib2 --truth anemoi-output.grib

    # Difference map for one field (forecast - truth), diverging colours centred on zero
    python scripts/compare_grib.py ... --plot 2t -o 2t-diff.png
    python scripts/compare_grib.py ... --plot z,500 -o z500-diff.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pygrib

# The DESCO interpreter drops the script's own directory from sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_grib import load_matrix, regrid_values, stored_values  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "data" / "regrid-0p25-to-n320.safetensors"
METADATA_DIR = ROOT / "data" / "quiet_grub" / "anemoi-metadata"

# The open-data files name the soil levels differently from the checkpoint; src/grib.rs
# SOIL_RENAMES, inverted here so both sides key on the model's names.
SOIL_RENAMES = {("sot", 1): "stl1", ("sot", 2): "stl2", ("vsw", 1): "swvl1", ("vsw", 2): "swvl2"}
ACCUMULATIONS = {"cp", "ro", "sf", "ssrd", "strd", "tp"}
CLOUD_COVERS = {"tcc", "hcc", "mcc", "lcc"}
WAVE_DIRECTION = "mwd"


def key_of(grb) -> tuple[str, int | None]:
    """(model variable name, pressure level or None) -- the identity both files share."""
    if grb.typeOfLevel == "isobaricInhPa":
        return grb.shortName, int(grb.level)
    if grb.typeOfLevel == "soilLayer":
        return SOIL_RENAMES.get((grb.shortName, int(grb.level)), grb.shortName), None
    return grb.shortName, None


def label(key: tuple[str, int | None]) -> str:
    name, level = key
    return f"{name}_{level}" if level is not None else name


def load_forecast(path: Path) -> dict:
    fields = {}
    with pygrib.open(str(path)) as grbs:
        for grb in grbs:
            fields[key_of(grb)] = stored_values(grb)
    return fields


def load_analysis(oper: Path, matrix, wanted: set) -> tuple[dict, set]:
    """The truth fields on N320, from the oper file and its wave sibling.

    Returns the fields and the keys whose message covers no time interval (endStep 0): an
    analysis has nothing to say about an accumulation, so those are not scored against it.

    Fields already on N320 -- an anemoi-inference output used as the reference -- are taken
    as stored; everything else is regridded from 0.25 degree.
    """
    fields, zero_step, steps = {}, set(), {}
    paths = [oper]
    wave = oper.with_name(oper.name.replace("oper", "wave"))
    if wave != oper and wave.exists():
        paths.append(wave)
    for path in paths:
        with pygrib.open(str(path)) as grbs:
            for grb in grbs:
                key = key_of(grb)
                # An anemoi output may carry the initial state too: keep the latest step.
                if key not in wanted or steps.get(key, -1) >= grb.endStep:
                    continue
                steps[key] = grb.endStep
                if grb.gridType == "reduced_gg" and grb.numberOfDataPoints == matrix.shape[0]:
                    values = stored_values(grb)
                else:
                    _, values, _ = regrid_values(grb, matrix)
                # The open-data cloud covers are in percent; the model works in fractions.
                if key[0] in CLOUD_COVERS and np.nanmax(values) > 1.5:
                    values = values / 100.0
                fields[key] = values
                if grb.endStep == 0:
                    zero_step.add(key)
                else:
                    zero_step.discard(key)
    return fields, zero_step


def difference(name: str, forecast: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = forecast - truth
    if name == WAVE_DIRECTION:
        diff = (diff + 180.0) % 360.0 - 180.0
    return diff


def score(name: str, forecast: np.ndarray, truth: np.ndarray) -> dict | None:
    both = np.isfinite(forecast) & np.isfinite(truth)
    if both.sum() < 2:
        return None
    f, t = forecast[both], truth[both]
    diff = difference(name, f, t)
    corr = float(np.corrcoef(f, t)[0, 1]) if f.std() > 0 and t.std() > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(diff.mean()),
        "corr": corr,
        "points": int(both.sum()),
        "truth_sd": float(t.std()),
    }


def plot_difference(name: str, level, diff: np.ndarray, out: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats = np.fromfile(METADATA_DIR / "latitudes.numpy", dtype="<f8")
    lons = np.fromfile(METADATA_DIR / "longitudes.numpy", dtype="<f8")
    finite = np.isfinite(diff)
    bound = float(np.percentile(np.abs(diff[finite]), 99)) if finite.any() else 1.0

    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    # A reduced Gaussian grid has no rectangle to mesh; one rasterised point per node.
    sc = ax.scatter(
        lons[finite], lats[finite], c=diff[finite], s=1.2, marker="s", linewidths=0,
        cmap="RdBu_r", vmin=-bound, vmax=bound, rasterized=True,
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    title = f"{name}{f' {level} hPa' if level is not None else ''}: forecast - truth"
    ax.set_title(title, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("forecast", type=Path, help="GRIB written by the Rust runtime (N320)")
    parser.add_argument("--truth", type=Path, required=True,
                        help="0.25 degree oper analysis valid at the forecast time")
    parser.add_argument("--baseline", type=Path,
                        help="0.25 degree oper analysis the forecast started from (persistence)")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--plot", help="variable to map as forecast - truth, e.g. 2t or z,500")
    parser.add_argument("-o", "--out", type=Path, help="PNG for --plot (default: <var>-diff.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    for path in (args.forecast, args.truth, args.baseline, args.matrix):
        if path is not None and not path.exists():
            sys.exit(f"error: {path} not found")

    matrix = load_matrix(args.matrix)
    forecast = load_forecast(args.forecast)
    wanted = set(forecast)
    truth, zero_step = load_analysis(args.truth, matrix, wanted)
    baseline = load_analysis(args.baseline, matrix, wanted)[0] if args.baseline else {}

    rows, unscored = [], []
    for key, f in forecast.items():
        name, level = key
        if key not in truth:
            unscored.append((label(key), "not in truth file"))
            continue
        if name in ACCUMULATIONS and key in zero_step:
            unscored.append((label(key), "accumulation; truth is an analysis (step 0)"))
            continue
        s = score(name, f, truth[key])
        if s is None:
            unscored.append((label(key), "no overlapping finite points"))
            continue
        if key in baseline:
            p = score(name, baseline[key], truth[key])
            s["persist"] = p["rmse"] if p else float("nan")
            s["skill"] = 1.0 - s["rmse"] / s["persist"] if s["persist"] > 0 else float("nan")
        rows.append((label(key), s))

    # Worst first: the diagnostic is at the top.
    if baseline:
        rows.sort(key=lambda r: (np.nan_to_num(r[1].get("skill", np.nan), nan=-np.inf)))
    else:
        rows.sort(key=lambda r: -r[1]["rmse"] / (r[1]["truth_sd"] or 1.0))

    header = (f"{'variable':<10} {'rmse':>11} {'bias':>11} {'corr':>7} "
              f"{'truth sd':>11} {'points':>8}")
    if baseline:
        header += f" {'persist':>11} {'skill':>7}"
    print(header)
    print("-" * len(header))
    for name, s in rows:
        line = (f"{name:<10} {s['rmse']:>11.4g} {s['bias']:>11.4g} {s['corr']:>7.3f} "
                f"{s['truth_sd']:>11.4g} {s['points']:>8}")
        if baseline:
            line += f" {s['persist']:>11.4g} {s['skill']:>7.3f}"
        print(line)

    if baseline:
        skills = np.array([s["skill"] for _, s in rows if np.isfinite(s.get("skill", np.nan))])
        beat = int((skills > 0).sum())
        print(f"\n{beat} of {skills.size} scored fields beat persistence; "
              f"median skill {np.median(skills):.3f}")
    if unscored:
        print("\nunscored:")
        for name, why in unscored:
            print(f"  {name:<10} {why}")

    if args.plot:
        name, _, level = args.plot.partition(",")
        key = (name, int(level) if level else None)
        if key not in forecast or key not in truth:
            sys.exit(f"error: {label(key)} is not in both files")
        diff = difference(name, forecast[key], truth[key])
        out = args.out or Path(f"{label(key)}-diff.png")
        plot_difference(name, key[1], diff, out, args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
