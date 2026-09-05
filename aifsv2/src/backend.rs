// The backend trait every module in this crate bounds on, in place of burn's, for the ops burn
// lacks.
//
// Two impls, matching the two ways a CubeCL backend is used:
//   CubeBackend  -- the raw backend; runs the kernel directly on CubeTensors.
//   Fusion<B>    -- what burn::backend::Cuda and Wgpu actually are. Fusion records ops lazily
//                   into a stream and replays them, so the op has to be registered as a stream
//                   entry whose execute() unwraps the primitives and calls the CubeBackend impl.
use std::marker::PhantomData;

use burn::{
    prelude::*,
    tensor::{
        TensorPrimitive,
        ops::{FloatTensor, IntTensor},
    },
};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_fusion::{
    Fusion, FusionBackend,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

use crate::scatter_max;

pub trait Backend: burn::tensor::backend::Backend {
    // tensor[indices[i], j, k] = max(tensor[indices[i], j, k], value[i, j, k]) along `dim`, for
    // every i. Duplicate indices fold in order, which is what makes it a segment reduction; the
    // identity is whatever `tensor` holds beforehand, so fill it with -inf, not zeros.
    fn float_scatter_max(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    // The same values in a freshly laid out row-major buffer; a no-op when already so. burn has
    // no tensor-level way to ask for this (reshape to the same shape is a no-op), and
    // burn-cubecl's flash attention reads its inputs as if they were contiguous, so the strided
    // views swap_dims produces have to be copied before they reach it (see attention_test.rs).
    fn float_contiguous(tensor: FloatTensor<Self>) -> FloatTensor<Self>;
}

// Tensor-level entry points, so callers never touch primitives.
pub fn contiguous<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::float_contiguous(
        tensor.into_primitive().tensor(),
    )))
}

pub fn scatter_max<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    indices: Tensor<B, 1, Int>,
    value: Tensor<B, D>,
) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::float_scatter_max(
        dim,
        tensor.into_primitive().tensor(),
        indices.into_primitive(),
        value.into_primitive().tensor(),
    )))
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Backend
    for CubeBackend<R, F, I, BT>
{
    fn float_scatter_max(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        scatter_max::select_assign_max(tensor, dim, indices, value)
    }

    fn float_contiguous(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        burn_cubecl::kernel::into_contiguous(tensor)
    }
}

impl<B: FusionBackend + Backend> Backend for Fusion<B> {
    fn float_scatter_max(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // `dim` lives on the op rather than in the IR: CustomOpIr only carries tensors, and the
        // stream always runs the very Operation object registered for a call, so it is not lost.
        #[derive(Debug)]
        struct ScatterMaxOp<B: FusionBackend> {
            desc: CustomOpIr,
            dim: usize,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend + Backend> Operation<B::FusionRuntime> for ScatterMaxOp<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let ([tensor, indices, value], [out]) = self.desc.as_fixed::<3, 1>();
                let tensor = handles.get_float_tensor::<B>(tensor);
                let indices = handles.get_int_tensor::<B>(indices);
                let value = handles.get_float_tensor::<B>(value);

                let output = B::float_scatter_max(self.dim, tensor, indices, value);

                handles.register_float_tensor::<B>(&out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices, &value]);
        let client = tensor.client.clone();
        let out = TensorIr::uninit(
            client.create_empty_handle(),
            tensor.shape.clone(),
            tensor.dtype,
        );
        let desc = CustomOpIr::new(
            "aifs::scatter_max",
            &[tensor.into_ir(), indices.into_ir(), value.into_ir()],
            &[out],
        );

        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                ScatterMaxOp::<B> {
                    desc,
                    dim,
                    _b: PhantomData,
                },
            )
            .pop()
            .expect("scatter_max registers exactly one output")
    }

    fn float_contiguous(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ContiguousOp<B: FusionBackend> {
            desc: CustomOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend + Backend> Operation<B::FusionRuntime> for ContiguousOp<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let ([tensor], [out]) = self.desc.as_fixed::<1, 1>();
                let tensor = handles.get_float_tensor::<B>(tensor);
                let output = B::float_contiguous(tensor);
                handles.register_float_tensor::<B>(&out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);
        let client = tensor.client.clone();
        let out = TensorIr::uninit(
            client.create_empty_handle(),
            tensor.shape.clone(),
            tensor.dtype,
        );
        let desc = CustomOpIr::new("aifs::contiguous", &[tensor.into_ir()], &[out]);

        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                ContiguousOp::<B> {
                    desc,
                    _b: PhantomData,
                },
            )
            .pop()
            .expect("contiguous registers exactly one output")
    }
}
