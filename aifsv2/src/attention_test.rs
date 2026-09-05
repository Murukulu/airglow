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
//! strides. The flash kernels honour them for key and value but not for the query: its reader
//! (cubek-attention global/simple/reader/query.rs) flattens each tile with `to_linear_slice()`
//! and walks it at a head_dim pitch, so rows within a tile are read from the neighbouring heads
//! instead. The answer is the right size and only partly correlated with the truth (#31). With
//! f32 inputs only the `unit` flash kernel can launch; blackbox_accelerated refuses f32 outright.
//!
//! The Swapped rows are asserted to FAIL, not skipped: if they start passing, upstream fixed the
//! reader and the copies in transformer.rs can go.
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
// strictly past the 4096 autotune threshold: tune.rs demotes the fallback only for seq_q > 4096,
// and at exactly 4096 it competes on speed and can win, in which case no flash kernel runs at all.
// A kernel that is right at one length and wrong at the other has a tail-tile bug; wrong at both,
// something more basic.
const SEQ_LENS: [usize; 2] = [4352, 4200];
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
            // Contiguous must agree; Swapped must not (the query misread, #31). transformer.rs
            // copies through backend::contiguous before the call, which the next two tests cover.
            let expect_ok = matches!(layout, Layout::Contiguous);
            if (rel <= TOL) != expect_ok {
                failures.push(format!("{layout:?} seq {seq}: rel rms {rel:.3e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "autotuned attention: Contiguous should match the fallback and Swapped should not: {failures:?}"
    );
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

// Which of the three inputs the flash kernels misread. Same swapped q/k/v, with either the query
// alone or key and value alone copied contiguous before the autotuned attention.
//
// The two sides take different readers in cubek-attention 0.2 (both routines, unit and
// blackbox_accelerated, share global::simple). Key and value are wrapped in AttentionGlobalLayout,
// which takes the tensor's strides, and read through it one vector at a time by cubek-matmul's
// FullStageGlobalReader, so a swapped view is addressed correctly. The query reader
// (global/simple/reader/query.rs) slices that same view but then calls `to_linear_slice()` --
// documented in cubecl-std as doing no check that the slice is contiguous -- and walks it with a
// row pitch from the tile geometry, i.e. head_dim, where a swapped view's rows sit heads *
// head_dim apart. So the query-only copy should agree with the fallback and the key/value-only
// copy should not. Both are asserted: if the first starts failing or the second starts passing,
// the upstream readers changed and the copies in transformer.rs should be looked at again (#31).
#[test]
fn only_the_query_needs_to_be_contiguous() {
    let device = Default::default();
    let mut failures = Vec::new();
    for (label, copy_q, copy_kv, expect_ok) in [
        ("q contiguous, k/v swapped", true, false, true),
        ("q swapped, k/v contiguous", false, true, false),
    ] {
        for seq in SEQ_LENS {
            let [q, k, v] = qkv::<TestBackend>(seq, Layout::Swapped, &device);
            let want = attention_fallback(q.clone(), k.clone(), v.clone(), None, None, options());
            let fix = |t, copy| if copy { crate::backend::contiguous(t) } else { t };
            let got = attention(fix(q, copy_q), fix(k, copy_kv), fix(v, copy_kv), None, None, options());
            let (max_abs, rel, corr) = metrics(got, want);
            let ok = rel <= TOL;
            let verdict = if ok { "ok  " } else { "FAIL" };
            println!("{label:<26} seq {seq:>5}  autotuned vs fallback   {verdict} max|d| {max_abs:.3e}  rel rms {rel:.3e}  corr {corr:.6}");
            if ok != expect_ok {
                let expected = if expect_ok { "agreement" } else { "disagreement" };
                failures.push(format!("{label} seq {seq}: rel rms {rel:.3e}, expected {expected}"));
            }
        }
    }
    assert!(failures.is_empty(), "flash attention's stride handling is no longer query-only: {failures:?}");
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
        // The fixed layout must agree and the raw one must not, as at the small sizes.
        if (rel <= TOL) != fix {
            failures.push(format!("{label}: rel rms {rel:.3e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "at grid size, Swapped+contiguous should match the fallback and Swapped should not: {failures:?}"
    );
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
                        // Every flash routine shares the query reader that misreads a swapped
                        // view, so any strategy that launches must agree on Contiguous and
                        // disagree on Swapped. In practice only `unit` launches for f32.
                        let expect_ok = matches!(layout, Layout::Contiguous);
                        if (rel <= TOL) != expect_ok {
                            failures.push(format!("{layout:?} seq {seq} {name}: rel rms {rel:.3e}"));
                        }
                    }
                    Err(err) => println!("{layout:<10?} seq {seq:>5}  {name:<30} unsupported: {err:?}"),
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "flash strategies: Contiguous should match the fallback and Swapped should not: {failures:?}"
    );
}
