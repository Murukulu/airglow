//! The one backend every *_test.rs runs on; build.rs picks it per machine (Cuda when the NVIDIA
//! driver is loaded, else Wgpu; `AIFS_TEST_BACKEND` overrides).
//!
//! Always a GPU backend, never ndarray: duplicate-index safety in the scatter-adds
//! (graph_tranformer_conv, sparse_segment_softmax, the prognostic residual) is a property of the
//! kernel, not of Burn. burn-ndarray accumulates duplicates correctly under every primitive with
//! one sequential host loop, so a CPU test would pass on an aggregation that is wrong on the
//! backend we ship.

#[cfg(test_backend = "cuda")]
pub type TestBackend = burn::backend::Cuda;

// burn::backend::Cuda minus the Fusion wrapper: the same kernels, launched one op at a time.
#[cfg(test_backend = "cuda-unfused")]
pub type TestBackend = burn_cubecl::CubeBackend<burn::cubecl::cuda::CudaRuntime, f32, i32, u8>;

#[cfg(not(any(test_backend = "cuda", test_backend = "cuda-unfused")))]
pub type TestBackend = burn::backend::wgpu::Wgpu;
