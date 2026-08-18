"""Unpickle an Anemoi inference .ckpt and hand back the AnemoiModelEncProcDec.

The checkpoint's top-level object is AnemoiModelInterface, which owns the
pre/post processors and, under `.model`, the AnemoiModelEncProcDec itself
(node_attributes, encoder, processor, decoder, boundings). Unpickling goes
through ckpt_to_safetensors._StubUnpickler so missing classes (flash_attn,
renamed anemoi layers) stub out instead of aborting the load.

Import it:
    from load_model import load_model
    model = load_model()                      # AnemoiModelEncProcDec

Or poke at it in a REPL, where `model` and `interface` are left in scope:
    .venv/bin/python -i scripts/load_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_to_safetensors import load_ckpt  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "data" / "aifs-single-mse-2.0.ckpt"


def load_interface(ckpt_path: Path = DEFAULT_CKPT):
    """The whole AnemoiModelInterface — processors included."""
    return load_ckpt(Path(ckpt_path))


def load_model(ckpt_path: Path = DEFAULT_CKPT):
    """Just the AnemoiModelEncProcDec."""
    return load_interface(ckpt_path).model


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    interface = load_interface(path)
    model = interface.model
    print(f"{type(model).__module__}.{type(model).__name__}")
    print(f"  submodules: {list(model._modules.keys())}")
    print(f"  attributes: {list(model.__dict__)}")
    print(f"  attributes: {interface.pre_processors.processors.normalizer._input_idx}")
    print(f"  attributes: {interface.pre_processors.processors.normalizer._output_idx}")
    print(f"  attributes: {interface.pre_processors.processors.normalizer._norm_mul}")
    # print(f"data_indices: {model.data_indices}")
    # print(f"  statistics: {model.statistics}")
    # print(f"  graph_data: {model._graph_data}")
    # print(f"  graph_data: {model._graph_data.node_types}")
