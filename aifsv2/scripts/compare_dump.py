"""Diff the Rust forward against the PyTorch reference, stage by stage.

Pairs each data/dump/<name>.<shape>.f32 that aifs.rs wrote with the
data/dump/ref_<name>.npy that ref_forward.py, ref_encoder.py, ref_processor.py
and ref_decoder.py wrote, in forward order, and stops at the first stage outside
tolerance: everything downstream of a bad stage is bad for the same reason, so
only the first failure is informative. The processor and decoder references are
computed from Rust's own dumps of their inputs, so a failure there is that
module's, not an upstream one carried through.

Per stage: max |a - b|, rms(a - b) / rms(b), and the correlation. Pre and
assemble are arithmetic and must agree to fp32 rounding; the network stages
reduce in a different order on the GPU and are allowed 1e-3 relative.

A failing stage is then broken down per channel, because that is what localises
the mistake. Each bad channel is named and, for the input side, tagged with its
normaliser mode and whether it is imputed; with the raw x dump present the error
is also split into points that were NaN in the input and points that were not.
A wrong head split breaks every channel; a wrong bounding, index scatter, or
processor order breaks a named few -- and the tags say which few have in common.
x_latent_data is additionally checked by column block against Rust's own pre_x,
since a swapped time-major flatten shows there and nowhere else.

    AIFS_DUMP_DIR=data/dump cargo run
    python scripts/compare_dump.py [--tol-net 1e-3] [--worst 10] [--all]
    python scripts/compare_dump.py --stats enc_x_latent     # one dump, both sides, in detail
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP_DIR = ROOT / "data" / "dump"
METADATA = ROOT / "data" / "quiet_grub" / "anemoi-metadata" / "ai-models.json"

# One processor block, in TransformerProcessorBlock.forward order. transformer.rs dumps `out`
# for every layer and the rest for the layers in TransformerProcessorConfig::dump_layers, under
# proc<layer>_; ref_processor.py writes the same names.
PROC_BLOCK = ["x_norm", "query", "key", "value",
              "query_rearranged", "key_rearranged", "value_rearranged",
              "attn", "proj", "attn_res", "mlp_norm", "mlp", "out"]


def proc_stages() -> list:
    """proc<layer>_* stages, layer by layer in forward order, for whatever reference dumps exist.
    A block's output gets a looser tolerance than its stages: it is the sum of them."""
    layers = set()
    for path in DUMP_DIR.glob("ref_proc[0-9]*_*.npy"):
        digits = path.name[len("ref_proc"):].split("_", 1)[0]
        layers.add(int(digits))
    return [(f"proc{layer}_{name}", 4e-3 if name == "out" else None, None)
            for layer in sorted(layers) for name in PROC_BLOCK
            if (DUMP_DIR / f"ref_proc{layer}_{name}.npy").exists()]


# Forward order: (name, relative-rms tolerance or None for --tol-net, channel naming side).
STAGES = [
    ("pre_x", 1e-5, "input"),
    ("x_latent_data", 1e-6, None),
    ("x_latent_hidden", 1e-6, None),
    # Inside the encoder, from ref_encoder.py; the Rust twins come from encoder.rs / block.rs.
    ("enc_x_src_emb", None, None),
    ("enc_x_dst_emb", None, None),
    ("enc_edge_attr", 1e-6, None),
    ("enc_x_src_norm", None, None),
    ("enc_x_dst_norm", None, None),
    ("enc_x_r", None, None),
    ("enc_query", None, None),
    ("enc_key", None, None),
    ("enc_value", None, None),
    ("enc_edges", None, None),
    ("enc_conv", None, None),
    ("enc_proj", None, None),
    ("enc_attn_out", None, None),
    ("enc_x_latent", None, None),
    # Inside the processor, from ref_processor.py; the Rust twins come from transformer.rs.
    *proc_stages(),
    ("proc_x_latent", 1e-2, None),
    # Inside the decoder, from ref_decoder.py, which is fed Rust's own proc_x_latent and
    # x_latent_data; the Rust twins come from decoder.rs / block.rs. So these judge the decoder
    # alone: proc_x_latent above is where the upstream drift is measured, and ref_decoder.py
    # prints how much the PyTorch decoder amplifies it.
    ("dec_edge_attr", 1e-6, None),
    ("dec_x_dst_emb", None, None),
    ("dec_conv", 3e-1, None),
    ("dec_proj", 3e-1, None),
    ("dec_attn_out", 3e-1, None),
    ("dec_block_out", 3e-1, None),
    ("dec_norm", 3e-1, None),
    ("dec_x_out", 3e-1, "output"),
    # End to end again, against ref_forward.py's hooked run: Rust's whole forward on its own
    # upstream, so these carry the processor drift through the decoder's gain and are expected to
    # sit well above --tol-net even when every stage above passes. They only say whether that
    # gain is what ref_decoder.py measured.
    ("bounded_x_out", 3e-1, "output"),
    ("post", 3e-1, "output"),
]


def read_rust(name: str) -> np.ndarray:
    (path,) = DUMP_DIR.glob(f"{name}.*.f32")
    shape = tuple(int(d) for d in path.name[len(name) + 1 : -len(".f32")].split("x"))
    return np.fromfile(path, np.float32).reshape(shape)


class Channels:
    """Variable names per model channel, plus how the checkpoint pre-processes each one."""

    def __init__(self):
        m = json.loads(METADATA.read_text())
        variables = m["dataset"]["variables"]
        idx = m["data_indices"]["data"]
        self.names = {
            "input": [variables[i] for i in idx["input"]["full"]],
            "output": [variables[i] for i in idx["output"]["full"]],
        }
        processors = m["config"]["data"]["processors"]
        normalizer = processors["normalizer"]["config"]
        self.default_mode = normalizer["default"]
        self.mode = {v: k for k in ("mean-std", "min-max", "max", "std", "none")
                     for v in normalizer.get(k) or []}
        self.imputed = {v for k, vs in processors["const_imputer"]["config"].items()
                        if isinstance(vs, list) for v in vs}

    def tag(self, side: str, c: int) -> str:
        name = self.names[side][c]
        if side != "input":
            return f"{name:<12}"
        mode = self.mode.get(name, self.default_mode)
        return f"{name:<12} {mode:<9}{' imputed' if name in self.imputed else '        '}"


def metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(max abs, relative rms, correlation) of a against reference b, over finite points."""
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    d = a - b
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    rms_b = float(np.sqrt(np.mean(b * b))) if b.size else 0.0
    rel = float(np.sqrt(np.mean(d * d))) / rms_b if rms_b > 0 else max_abs
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size and a.std() > 0 and b.std() > 0 else float("nan")
    return max_abs, rel, corr


def check(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if a.shape != b.shape:
        print(f"{name:<16} FAIL shape {a.shape} vs ref {b.shape}")
        return False
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if (nan_a != nan_b).any():
        print(f"{name:<16} FAIL NaN at {int(nan_a.sum())} points, ref at {int(nan_b.sum())}, "
              f"{int((nan_a != nan_b).sum())} disagree")
        return False
    max_abs, rel, corr = metrics(a, b)
    ok = rel <= tol
    print(f"{name:<16} {'ok  ' if ok else 'FAIL'} max|d| {max_abs:.3e}  rel rms {rel:.3e}  "
          f"corr {corr:.6f}  (tol {tol:.0e})")
    return ok


def check_blocks(x_latent_data: np.ndarray, pre_x: np.ndarray) -> None:
    """The Rust flatten order, against Rust's own pre_x: time is the outer index of the channel axis."""
    _, time, _, vars_ = pre_x.shape
    for t in range(time):
        block = x_latent_data[:, t * vars_ : (t + 1) * vars_]
        same = np.array_equal(block, pre_x[0, t], equal_nan=True)
        print(f"  cols {t * vars_}:{(t + 1) * vars_} == pre_x[0, {t}]: {'ok' if same else 'MISMATCH'}")
    extra = x_latent_data.shape[1] - time * vars_
    print(f"  cols {time * vars_}:{x_latent_data.shape[1]}: {extra} node attributes (latlons + trainable)")


def describe(values: np.ndarray) -> str:
    """A constant prints as itself; anything else as its range."""
    if values.size == 0:
        return "-"
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return f"{lo:.6g}" if lo == hi else f"[{lo:.4g}, {hi:.4g}]"


def check_channels(a, b, side: str, channels: Channels, tol: float, worst: int, raw=None) -> None:
    """Per channel over the last axis, every leading axis flattened; pre_x reports per (time, channel)."""
    a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    raw2 = raw.reshape(-1, raw.shape[-1]) if raw is not None else None
    rows = []
    for c in range(a2.shape[1]):
        max_abs, rel, corr = metrics(a2[:, c], b2[:, c])
        rows.append((rel, c, max_abs, corr))
    rows.sort(reverse=True)
    bad = sum(r[0] > tol for r in rows)
    print(f"  {bad} of {len(rows)} channels over tol; worst {min(worst, len(rows))}:")
    for rel, c, max_abs, corr in rows[:worst]:
        line = f"    {c:>3} {channels.tag(side, c)} rel rms {rel:.3e}  max|d| {max_abs:.3e}  corr {corr:.6f}"
        if raw2 is not None:
            # Where the input was NaN the imputer decided the value; elsewhere only the normaliser did.
            nan = np.isnan(raw2[:, c])
            finite_err = float(np.abs(a2[~nan, c] - b2[~nan, c]).max()) if (~nan).any() else 0.0
            line += (f"\n        input NaN at {int(nan.sum())} pts: rust {describe(a2[nan, c])}, "
                     f"ref {describe(b2[nan, c])}; elsewhere max|d| {finite_err:.3e}")
        print(line)


def stats(name: str) -> None:
    """Both sides of one dump on their own terms, then against each other.

    The per-side rows say which side is the outlier: an unnormalised max on one
    side is a missed LayerNorm or embedding. The sorted comparisons say whether
    the numbers are right but in the wrong place: rows sorted independently
    agree while the raw tensors do not when the rows (nodes) are permuted,
    columns likewise for the channels; if the sorted values differ too, the
    arithmetic itself is wrong.
    """
    if next(DUMP_DIR.glob(f"{name}.*.f32"), None) is None:
        print(f"{name}: no rust dump in {DUMP_DIR}")
        return
    a = read_rust(name)
    ref_path = DUMP_DIR / f"ref_{name}.npy"
    b = np.load(ref_path) if ref_path.exists() else None
    if b is None:
        print(f"{name}: no ref dump, rust side only")

    print(f"{name}  rust {list(a.shape)}  ref {list(b.shape) if b is not None else '-'}")
    print(f"  {'':<6} {'mean':>11} {'std':>11} {'min':>11} {'max':>11} {'max|.|':>11} {'NaN':>9}")
    for label, t in (("rust", a), ("ref", b)):
        if t is None:
            continue
        f = t[np.isfinite(t)].astype(np.float64)
        print(f"  {label:<6} {f.mean():11.4e} {f.std():11.4e} {f.min():11.4e} {f.max():11.4e} "
              f"{np.abs(f).max():11.4e} {int(np.isnan(t).sum()):9d}")
    if b is None or a.shape != b.shape:
        return

    max_abs, rel, corr = metrics(a, b)
    print(f"  raw            max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6f}")
    a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    for label, axis in (("rows sorted   ", 0), ("columns sorted", 1)):
        max_abs, rel, corr = metrics(np.sort(a2, axis=axis), np.sort(b2, axis=axis))
        print(f"  {label} max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tol-net", type=float, default=1e-3, help="relative rms for the network stages")
    parser.add_argument("--worst", type=int, default=10, help="channels to list in a per-channel breakdown")
    parser.add_argument("--all", action="store_true", help="keep going past the first failure, break down every stage")
    parser.add_argument("--stats", metavar="NAME", help="per-side statistics and permutation test for one dump, then exit")
    parser.add_argument("--continue-from", metavar="NAME",
                        help="start at this stage and check forward (stages already verified upstream are skipped)")
    args = parser.parse_args()

    if args.stats:
        stats(args.stats)
        return 0

    stages = STAGES
    if args.continue_from:
        names = [name for name, _, _ in STAGES]
        if args.continue_from not in names:
            parser.error(f"unknown stage {args.continue_from!r}; choose one of: {', '.join(names)}")
        stages = STAGES[names.index(args.continue_from):]

    channels = Channels()
    try:
        raw_x = read_rust("x")
    except ValueError:
        raw_x = None
    pre_x = None
    for name, tol, side in stages:
        tol = args.tol_net if tol is None else tol
        rust_path = next(DUMP_DIR.glob(f"{name}.*.f32"), None)
        ref_path = DUMP_DIR / f"ref_{name}.npy"
        missing = [side for side, present in (("rust", rust_path), ("ref", ref_path.exists())) if not present]
        if missing:
            print(f"{name:<16} skip: no {' or '.join(missing)} dump, not comparing")
            continue
        a, b = read_rust(name), np.load(ref_path)

        ok = check(name, a, b, tol)
        if name == "pre_x":
            pre_x = a
        if name == "x_latent_data" and pre_x is None and next(DUMP_DIR.glob("pre_x.*.f32"), None) is not None:
            pre_x = read_rust("pre_x")
        if name == "x_latent_data" and pre_x is not None:
            check_blocks(a, pre_x)
        if side and (not ok or args.all):
            check_channels(a, b, side, channels, tol, args.worst, raw=raw_x if name == "pre_x" else None)
        if not ok and not args.all:
            print(f"\nfirst divergence: {name}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
