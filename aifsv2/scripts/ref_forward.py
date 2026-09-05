"""PyTorch reference for the Rust forward, captured at the same stages aifs.rs dumps.

Reads the physical-units input the Rust binary wrote (data/dump/x.*.f32), runs
anemoi's predict_step on the real checkpoint with forward hooks at the module
boundaries, and writes each capture as data/dump/ref_<name>.npy in the Rust
layout -- anemoi's ensemble axis squeezed, the latent skip added -- so every
file pairs with its Rust twin of the same <name>.

Needs the reference venv (anemoi-models 0.9.3 installed for real) and a GPU:
    python scripts/ref_forward.py [--window]

--window keeps the checkpoint's 1120-node sliding window. The default drops it,
because the Rust processor does not window yet (transformer.rs).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_model import load_interface, use_sdpa  # noqa: E402

DUMP_DIR = Path(__file__).resolve().parent.parent / "data" / "dump"


def read_dump(name: str) -> torch.Tensor:
    """<name>.<d0>x<d1>x...f32: raw little-endian f32, as debug.rs writes it."""
    (path,) = DUMP_DIR.glob(f"{name}.*.f32")
    shape = tuple(int(d) for d in path.name[len(name) + 1 : -len(".f32")].split("x"))
    return torch.from_numpy(np.fromfile(path, np.float32).reshape(shape))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--window", action="store_true", help="keep the sliding-window attention")
    args = parser.parse_args()

    x = read_dump("x").cuda()  # [batch, time, grid, vars], physical units, NaN where unset
    interface = use_sdpa(load_interface(), window=args.window).cuda().eval()
    m = interface.model
    assert len(m.boundings), "no boundings: bounded_x_out would need the residual by hand"

    caps = {}

    def cap(name):
        def hook(_, inp, out):
            caps[name] = (inp, out)

        return hook

    interface.pre_processors.register_forward_hook(cap("pre"))
    m.encoder.register_forward_hook(cap("encoder"))
    m.processor.register_forward_hook(cap("processor"))
    m.decoder.register_forward_hook(cap("decoder"))
    m.boundings[-1].register_forward_hook(cap("bounding"))
    interface.post_processors.register_forward_hook(cap("post"))

    with torch.inference_mode():
        interface.predict_step(x)

    # anemoi carries an ensemble axis Rust does not: [batch, time, ensemble, grid, vars] through
    # pre, [batch, ensemble, grid, vars] out of the boundings and post. The mappers see the same
    # [(batch grid), channels] as Rust. _assemble_input/_assemble_output are methods, not modules,
    # so their tensors are read off the encoder's arguments and the last bounding's output.
    #
    # The decoder capture is written as dec_x_out_hooked, not dec_x_out: Rust's dec_x_out pairs
    # with ref_decoder.py's decoder-alone run (fed Rust's own proc_x_latent), and this end-to-end
    # capture is what that run checks itself against.
    (enc_in,), enc_out = caps["encoder"]
    (proc_in,), proc_out = caps["processor"]
    ref = {
        "pre_x": caps["pre"][1].squeeze(2),
        "x_latent_data": enc_in[0],
        "x_latent_hidden": enc_in[1],
        "enc_x_latent": enc_out[1],
        "proc_x_latent": proc_out + proc_in,  # latent_skip is added outside the processor
        "dec_x_out_hooked": caps["decoder"][1],
        "bounded_x_out": caps["bounding"][1].flatten(0, 2),
        "post": caps["post"][1].flatten(0, 2),
    }
    for name, t in ref.items():
        np.save(DUMP_DIR / f"ref_{name}.npy", t.float().cpu().numpy())
        print(f"ref_{name} {tuple(t.shape)}")


if __name__ == "__main__":
    main()
