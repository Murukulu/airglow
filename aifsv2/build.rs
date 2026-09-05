//! Chooses the GPU backend the unit tests run on; src/test_backend.rs turns the `test_backend`
//! cfg this emits into a type.
//!
//! `AIFS_TEST_BACKEND=cuda|cuda-unfused|wgpu` wins outright. Otherwise Cuda when the NVIDIA
//! kernel driver is loaded, else Wgpu: the datacenter hosts run the headless driver, which has
//! no Vulkan ICD for wgpu to find an adapter through, and Cuda is the backend main.rs ships.
//!
//! Cargo re-runs this only when this file or the variable changes, not when the hardware does;
//! after moving a checkout between machine types, `touch build.rs`.

use std::{env, path::Path};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=AIFS_TEST_BACKEND");
    println!(
        "cargo::rustc-check-cfg=cfg(test_backend, values(\"cuda\", \"cuda-unfused\", \"wgpu\"))"
    );

    // cuda-unfused is the same CubeCL kernels without the Fusion pass in front of them, for
    // telling a fused-kernel miscompile apart from a kernel bug.
    let backend = match env::var("AIFS_TEST_BACKEND").ok().as_deref() {
        Some("cuda") => "cuda",
        Some("cuda-unfused") => "cuda-unfused",
        Some("wgpu") => "wgpu",
        Some(other) => {
            panic!("AIFS_TEST_BACKEND must be `cuda`, `cuda-unfused` or `wgpu`, got `{other}`")
        }
        None if nvidia_driver_loaded() => "cuda",
        None => "wgpu",
    };
    println!("cargo::rustc-cfg=test_backend=\"{backend}\"");
}

// Either node exists iff the NVIDIA kernel module is loaded. Checked rather than `nvidia-smi` so
// a missing toolkit on PATH does not misroute a GPU host to wgpu.
fn nvidia_driver_loaded() -> bool {
    Path::new("/proc/driver/nvidia/version").exists() || Path::new("/dev/nvidiactl").exists()
}
