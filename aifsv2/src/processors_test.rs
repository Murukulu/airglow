use burn::tensor::Tolerance;

use super::*;

type TestBackend = burn::backend::wgpu::Wgpu;
type TestDevice = Device<TestBackend>;

// Three input channels, four output channels. Input channel 0 is filled but has no output, input
// channel 2 is filled and restores into output channel 3, and input channel 1 is not imputed at
// all. That is the shape of the real asymmetry -- forcings drop out, diagnostics appear -- and
// coping with it is why the imputer carries three lists rather than one.
fn imputer(device: &TestDevice) -> Imputer<TestBackend> {
    Imputer {
        fill: bools(vec![true, false, true], device),
        restore_from: indices(vec![0, 0, 0, 2], device),
        restores: bools(vec![false, false, false, true], device),
    }
}

#[test]
fn imputer_fills_only_nan_and_only_its_channels() {
    let device = Default::default();
    // [1, 1, 2, 3]: two grid points, three channels, NaN in channels 0 and 1.
    let x = Tensor::<TestBackend, 4>::from_floats(
        [[[[f32::NAN, f32::NAN, 5.0], [1.0, 2.0, f32::NAN]]]],
        &device,
    );

    let (out, _) = imputer(&device).forward(x);
    let values = out.into_data().to_vec::<f32>().unwrap();

    assert_eq!(values[0], 0.0, "channel 0 is in the fill list");
    assert!(values[1].is_nan(), "channel 1 is not");
    assert_eq!(values[2], 5.0, "a written value is left alone");
    assert_eq!(values[3..5], [1.0, 2.0]);
    assert_eq!(values[5], 0.0, "channel 2 is filled at the second point");
}

// The mask is recorded against input channels and consumed against output channels. Confusing the
// two is invisible until one output column is silently wrong, so this pins the crossover: input
// channel 2 must land on output channel 3 and nowhere else.
#[test]
fn imputer_round_trips_nan_from_input_to_output_channels() {
    let device = Default::default();
    let x =
        Tensor::<TestBackend, 4>::from_floats([[[[0.0, 0.0, f32::NAN], [0.0, 0.0, 7.0]]]], &device);

    let (_, imputed) = imputer(&device).forward(x);
    assert_eq!(imputed.dims(), [2, 4], "one row per grid point, output width");

    // The model's output: two grid points, four channels, all finite.
    let y =
        Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device);
    let restored = y
        .mask_fill(imputed, f32::NAN)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert!(restored[3].is_nan(), "grid point 0 was imputed");
    assert_eq!(restored[0..3], [1.0, 2.0, 3.0], "other channels untouched");
    assert_eq!(restored[7], 8.0, "grid point 1 held real data");
}

// Only the first timestep is recorded (imputer.py:227). A NaN appearing at a later timestep must
// not reach the mask, or the inverse blanks a point that had data.
#[test]
fn imputer_records_the_first_timestep_only() {
    let device = Default::default();
    // [1, 2, 1, 3]: two timesteps, one grid point. Channel 2 is NaN at t=1 only.
    let x =
        Tensor::<TestBackend, 4>::from_floats([[[[0.0, 0.0, 9.0]], [[0.0, 0.0, f32::NAN]]]], &device);

    // Read back through int: wgpu stores Bool as U32, which to_vec::<bool> will not accept.
    let (_, imputed) = imputer(&device).forward(x);
    assert_eq!(
        imputed.int().into_data().to_vec::<i32>().unwrap(),
        [0, 0, 0, 0]
    );
}

#[test]
fn normalizer_inverts_itself() {
    let device: TestDevice = Default::default();
    let normalizer = Normalizer::<TestBackend> {
        input_mul: Tensor::from_floats([2.0, 0.5], &device),
        input_add: Tensor::from_floats([-1.0, 3.0], &device),
        output_mul: Tensor::from_floats([2.0, 0.5], &device),
        output_add: Tensor::from_floats([-1.0, 3.0], &device),
    };

    let x = Tensor::<TestBackend, 4>::from_floats([[[[4.0, 8.0], [-2.0, 0.0]]]], &device);
    let normalised = normalizer.forward(x.clone());
    assert_eq!(
        normalised.clone().into_data().to_vec::<f32>().unwrap(),
        [7.0, 7.0, -5.0, 3.0],
        "x * mul + add, broadcast per channel"
    );

    // The inverse consumes the [N, vars] output layout, so reshape rather than re-deriving.
    normalizer
        .inverse(normalised.reshape([2, 2]))
        .into_data()
        .assert_approx_eq::<f32>(&x.reshape([2, 2]).into_data(), Tolerance::default());
}

// The reference variable's NaN drives the targets; a finite reference leaves them alone.
#[test]
fn conditional_nan_follows_its_reference() {
    let device: TestDevice = Default::default();
    let conditional_nan = ConditionalNan::<TestBackend> {
        reference: indices(vec![0; 4], &device),
        targets: bools(vec![false, true, true, false], &device),
    };

    let y = Tensor::<TestBackend, 2>::from_floats(
        [[f32::NAN, 1.0, 2.0, 3.0], [9.0, 1.0, 2.0, 3.0]],
        &device,
    );
    let out = conditional_nan
        .inverse(y)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert!(out[1].is_nan() && out[2].is_nan(), "reference is NaN");
    assert_eq!(out[3], 3.0, "channel 3 is not a target");
    assert_eq!(
        out[5..8],
        [1.0, 2.0, 3.0],
        "reference is finite, nothing changes"
    );
}
