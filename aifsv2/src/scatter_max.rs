// select_assign with max in place of add, which burn does not have.

use burn::cubecl::{CubeDim, calculate_cube_count_elemwise, cube, prelude::*};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use cubecl::std::{FastDivmod, tensor::layout::linear::LinearView};

pub(crate) fn select_assign_max<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    axis: usize,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    // One thread per non-axis coordinate; each walks the whole axis serially, which is what makes
    // duplicate indices safe without atomics.
    let working_units = value.meta.num_elements() / value.meta.shape()[axis];
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let mut shape = SequenceArg::new();
    for dim in value.meta.shape().iter() {
        shape.push(*dim);
    }

    let address_type = [&tensor, &indices, &value]
        .into_iter()
        .map(|t| t.required_address_type())
        .max()
        .unwrap_or_default();

    select_max_kernel::launch(
        &tensor.client,
        cube_count,
        cube_dim,
        address_type,
        tensor.clone().into_tensor_arg(),
        indices.into_linear_view(),
        value.into_tensor_arg(),
        shape,
        working_units,
        axis,
        [tensor_dtype.into(), indices_dtype.into()],
    );
    tensor
}

#[cube(launch, address_type = "dynamic")]
fn select_max_kernel<F: Numeric, I: Numeric>(
    tensor: &mut Tensor<F>,
    indices: &LinearView<I>,
    value: &Tensor<F>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    // Unravel this thread's linear id over the non-axis dims, outermost last, into a base offset
    // in each tensor.
    let rank = value_shape.len().comptime();
    let mut offset = ABSOLUTE_POS;
    let mut offset_tensor = 0;
    let mut offset_value = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != axis {
            let (rem, local_pos) = value_shape[i].div_mod(offset);
            offset = rem;

            offset_tensor += local_pos * tensor.stride(i);
            offset_value += local_pos * value.stride(i);
        }
    }

    let strides_tensor_dim = tensor.stride(axis);
    let strides_value_dim = value.stride(axis);

    for i in 0..value.shape(axis) {
        let index_tensor = usize::cast_from(indices[i]) * strides_tensor_dim + offset_tensor;
        let index_value = i * strides_value_dim + offset_value;

        tensor[index_tensor] = clamp_min(tensor[index_tensor], value[index_value]);
    }
}
