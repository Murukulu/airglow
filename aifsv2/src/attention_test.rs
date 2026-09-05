//! Does burn's fused attention agree with its own naive fallback on this GPU?
//!
//! transformer.rs calls burn::tensor::module::attention, which on CubeCL autotunes between a
//! naive fallback and several flash-attention variants (burn-cubecl kernel/attention/tune.rs).
//! Above seq_q = 4096 the fallback is demoted and a flash variant wins on speed alone; nothing
//! checks its answer. The attention output diverging from PyTorch while q/k/v agree is what a
//! wrong winner would look like, so this pins each candidate against the fallback directly.
//!
//! Shapes follow the processor: [1, 16 heads, S, 64]. S sits just past the 4096 threshold so the
//! autotuner takes the branch production takes, while the fallback's [16, S, S] score matrix
//! stays around 1 GB rather than the 97 GB it would be at 40,320 nodes.
//!
//! Each case runs in two memory layouts, because MultiHeadSelfAttention::forward does not hand
//! the kernel a contiguous [1, H, S, D]: it reshapes lin_q's [1, S, H*D] and swap_dims(1, 2), so
//! the kernel sees a view whose memory is [S, H, D]. The fallback goes through matmul and honours
//! strides; a flash kernel that assumes contiguity would read heads and rows interleaved and give
//! an answer of the right size that is only partly correlated with the truth.
use burn::{
    prelude::*,
    tensor::{
        Distribution, TensorPrimitive,
        module::{attention, attention_fallback},
        ops::AttentionModuleOptions,
    },
};
use burn_cubecl::kernel::attention::{AttentionStrategy, attention as attention_with};
use cubek::attention::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;

use crate::test_backend::TestBackend;

// The strategy-level test needs CubeTensors, so it runs on the unfused backend whatever the
// suite's TestBackend is (Fusion's primitives are lazy placeholders).
#[cfg(any(test_backend = "cuda", test_backend = "cuda-unfused"))]
type Raw = burn_cubecl::CubeBackend<burn::cubecl::cuda::CudaRuntime, f32, i32, u8>;
#[cfg(not(any(test_backend = "cuda", test_backend = "cuda-unfused")))]
type Raw = burn_cubecl::CubeBackend<burn::cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
// Aligned and unaligned sequence lengths either side of the tile sizes flash kernels use, both
// past the 4096 autotune threshold. A kernel that is right at one and wrong at the other has a
// tail-tile bug; wrong at both, something more basic.
const SEQ_LENS: [usize; 2] = [4096, 4200];
const TOL: f64 = 1e-3;

fn options() -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    }
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    // Fresh [1, H, S, D] tensors, as a kernel test would naturally build them.
    Contiguous,
    // What transformer.rs passes: [1, S, H, D] in memory, viewed as [1, H, S, D] by swap_dims.
    Swapped,
}

fn qkv<B: Backend>(seq: usize, layout: Layout, device: &B::Device) -> [Tensor<B, 4>; 3] {
    // Seeded so a failure reproduces; unit scale keeps the logits O(sqrt(D)), like the real
    // layer's after LayerNorm, so the softmax is neither flat nor one-hot.
    B::seed(device, 0);
    let make = || match layout {
        Layout::Contiguous => {
            Tensor::random([1, HEADS, seq, HEAD_DIM], Distribution::Normal(0.0, 1.0), device)
        }
        Layout::Swapped => {
            Tensor::random([1, seq, HEADS, HEAD_DIM], Distribution::Normal(0.0, 1.0), device)
                .swap_dims(1, 2)
        }
    };
    [make(), make(), make()]
}

// attention_fallback over query rows CHUNK at a time. Each query row's softmax is over the keys
// alone, so splitting the query axis is exact; it bounds the score matrix at [16, CHUNK, S].
fn chunked_fallback<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    chunk: usize,
) -> Tensor<B, 4> {
    let [b, h, seq, _] = q.shape().dims();
    let parts = (0..seq)
        .step_by(chunk)
        .map(|start| {
            let end = (start + chunk).min(seq);
            let q = q.clone().slice([0..b, 0..h, start..end]);
            attention_fallback(q, k.clone(), v.clone(), None, None, options())
        })
        .collect();
    Tensor::cat(parts, 2)
}

// (max |a - b|, rms(a - b) / rms(b), correlation) over the whole tensor, in f64.
fn metrics<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> (f64, f64, f64) {
    let a: Vec<f64> = a.into_data().to_vec::<f32>().unwrap().iter().map(|&v| v as f64).collect();
    let b: Vec<f64> = b.into_data().to_vec::<f32>().unwrap().iter().map(|&v| v as f64).collect();
    let n = a.len() as f64;
    let max_abs = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let rms_d = (a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / n).sqrt();
    let rms_b = (b.iter().map(|y| y * y).sum::<f64>() / n).sqrt();
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>();
    let (va, vb) = (
        a.iter().map(|x| (x - ma).powi(2)).sum::<f64>(),
        b.iter().map(|y| (y - mb).powi(2)).sum::<f64>(),
    );
    (max_abs, rms_d / rms_b, cov / (va * vb).sqrt())
}

// What production runs: the autotuned attention, against the naive fallback, on TestBackend.
#[test]
fn autotuned_attention_matches_fallback() {
    let device = Default::default();
    let mut failures = Vec::new();
    for layout in [Layout::Contiguous, Layout::Swapped] {
        for seq in SEQ_LENS {
            let [q, k, v] = qkv::<TestBackend>(seq, layout, &device);
            let want = attention_fallback(q.clone(), k.clone(), v.clone(), None, None, options());
            let got = attention(q, k, v, None, None, options());
            let (max_abs, rel, corr) = metrics(got, want);
            let verdict = if rel <= TOL { "ok  " } else { "FAIL" };
            println!("{layout:<10?} seq {seq:>5}  autotuned vs fallback           {verdict} max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6}");
            // The Swapped rows document the upstream bug; transformer.rs routes around it with
            // backend::contiguous, which the next two tests cover.
            if matches!(layout, Layout::Contiguous) && !(rel <= TOL) {
                failures.push(format!("{layout:?} seq {seq}: rel rms {rel:.3e}"));
            }
        }
    }
    assert!(failures.is_empty(), "autotuned attention disagrees with fallback: {failures:?}");
}

// The route transformer.rs takes: a swapped view made contiguous through the crate backend
// trait, then the autotuned attention.
#[test]
fn attention_on_contiguous_copy_of_swapped_view_matches_fallback() {
    let device = Default::default();
    let mut failures = Vec::new();
    for seq in SEQ_LENS {
        let [q, k, v] = qkv::<TestBackend>(seq, Layout::Swapped, &device);
        let want = attention_fallback(q.clone(), k.clone(), v.clone(), None, None, options());
        let [q, k, v] = [q, k, v].map(crate::backend::contiguous);
        let got = attention(q, k, v, None, None, options());
        let (max_abs, rel, corr) = metrics(got, want);
        let verdict = if rel <= TOL { "ok  " } else { "FAIL" };
        println!("Swapped+contiguous seq {seq:>5}  autotuned vs fallback   {verdict} max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6}");
        if !(rel <= TOL) {
            failures.push(format!("seq {seq}: rel rms {rel:.3e}"));
        }
    }
    assert!(failures.is_empty(), "attention on contiguous inputs disagrees with fallback: {failures:?}");
}

// The production shape and entry point against the exact answer, in the raw swapped layout and
// after backend::contiguous, as MultiHeadSelfAttention::forward does. Ignored by
// default: the chunked fallback touches [16, 1024, 40320] scores per chunk, some GB of scratch,
// and the whole thing takes minutes. Run with `cargo test attention_test -- --ignored --nocapture`.
#[test]
#[ignore]
fn autotuned_attention_matches_chunked_fallback_at_grid_size() {
    const GRID: usize = 40320;
    let device = Default::default();
    let mut failures = Vec::new();
    for (label, fix) in [("Swapped", false), ("Swapped+contiguous", true)] {
        let [q, k, v] = qkv::<TestBackend>(GRID, Layout::Swapped, &device);
        let want = chunked_fallback(q.clone(), k.clone(), v.clone(), 1024);
        let [q, k, v] = if fix {
            [q, k, v].map(crate::backend::contiguous)
        } else {
            [q, k, v]
        };
        let got = attention(q, k, v, None, None, options());
        let (max_abs, rel, corr) = metrics(got, want);
        let verdict = if rel <= TOL { "ok  " } else { "FAIL" };
        println!("{label:<18} seq {GRID:>5}  autotuned vs chunked fallback   {verdict} max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6}");
        if fix && !(rel <= TOL) {
            failures.push(format!("{label}: rel rms {rel:.3e}"));
        }
    }
    // Only the fixed layout has to pass; the raw one is printed to show the bug is still there.
    assert!(failures.is_empty(), "attention on contiguous inputs disagrees with fallback at grid size: {failures:?}");
}

// Every candidate the autotuner can pick, named, against the fallback, on the raw CubeBackend.
// A strategy the kernel refuses for this shape is reported, not failed: the tuner skips those too.
#[test]
fn every_attention_strategy_matches_fallback() {
    let device = Default::default();
    let strategies = || {
        let mut s = vec![("unit".to_string(), AttentionStrategy::FlashUnit)];
        // tune.rs registers these with seq_q = seq_kv = 1 and num_planes in [2, 4, 8].
        for num_planes in [2u8, 4, 8] {
            s.push((
                format!("blackbox_accelerated_{num_planes}_planes"),
                AttentionStrategy::FlashBlackboxAccelerated(BlackboxAcceleratedStrategy {
                    num_planes,
                    seq_q: 1,
                    seq_kv: 1,
                }),
            ));
        }
        s
    };
    let run = |q: &Tensor<Raw, 4>, k: &Tensor<Raw, 4>, v: &Tensor<Raw, 4>, strategy| {
        attention_with(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            None,
            None,
            options(),
            strategy,
        )
        .map(|out| Tensor::<Raw, 4>::from_primitive(TensorPrimitive::Float(out)))
    };

    let mut failures = Vec::new();
    for layout in [Layout::Contiguous, Layout::Swapped] {
        for seq in SEQ_LENS {
            let [q, k, v] = qkv::<Raw>(seq, layout, &device);
            let want = run(&q, &k, &v, AttentionStrategy::Fallback).expect("fallback always runs");
            for (name, strategy) in strategies() {
                match run(&q, &k, &v, strategy) {
                    Ok(got) => {
                        let (max_abs, rel, corr) = metrics(got, want.clone());
                        let verdict = if rel <= TOL { "ok  " } else { "FAIL" };
                        println!("{layout:<10?} seq {seq:>5}  {name:<30} {verdict} max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6}");
                        // Swapped rows are informational: the unit kernel ignores strides, and
                        // transformer.rs hands it none.
                        if matches!(layout, Layout::Contiguous) && !(rel <= TOL) {
                            failures.push(format!("{layout:?} seq {seq} {name}: rel rms {rel:.3e}"));
                        }
                    }
                    Err(err) => println!("{layout:<10?} seq {seq:>5}  {name:<30} unsupported: {err:?}"),
                }
            }
        }
    }
    assert!(failures.is_empty(), "flash attention strategies disagree with fallback: {failures:?}");
}
