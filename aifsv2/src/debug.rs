use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use burn::prelude::*;

use crate::backend::Backend;

/// The directory `AIFS_DUMP_DIR` names; None, and no dump is written, when it is unset.
fn dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("AIFS_DUMP_DIR").map(PathBuf::from))
        .as_deref()
}

/// Write a tensor to `$AIFS_DUMP_DIR/<name>.<d0>x<d1>x...f32` as raw little-endian f32; nothing
/// happens when the variable is unset.
///
/// The shape rides in the filename so the reader needs nothing else:
/// `np.fromfile(path, np.float32).reshape(shape)`. scripts/compare_dump.py pairs each file with
/// the `ref_<name>.npy` that the scripts/ref_*.py write from the PyTorch side, so a name used here
/// must be produced there under the same name and layout.
///
///     AIFS_DUMP_DIR=data/dump cargo run
pub fn dump<B: Backend, const D: usize>(name: &str, t: &Tensor<B, D>) {
    let Some(dir) = dir() else { return };
    let shape = t
        .shape()
        .dims::<D>()
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    let bytes = t
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(dir.join(format!("{name}.{shape}.f32")), bytes).unwrap();
}
