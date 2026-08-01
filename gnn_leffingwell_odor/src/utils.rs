use burn::{
    prelude::*,
    tensor::{TensorMetadata, ops::FloatTensor},
};

/// Reshape a tensor by prepending size-1 dimensions until it has `target_ndims` dimensions.
pub fn unsqueeze_leading<B: Backend>(
    tensor: FloatTensor<B>,
    target_ndims: usize,
) -> FloatTensor<B> {
    let shape = tensor.shape();
    let ndims = shape.num_dims();
    if ndims >= target_ndims {
        return tensor;
    }
    let mut new_dims = vec![1usize; target_ndims - ndims];
    for i in 0..ndims {
        new_dims.push(shape[i]);
    }
    B::float_reshape(tensor, Shape::from(new_dims))
}

// Perform scatter mean based on batch.
pub fn scatter_mean<B: Backend>(
    x: Tensor<B, 2>,         // [N, F]
    idxs: Tensor<B, 1, Int>, // [N]
    num_graphs: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [n, f] = x.dims();
    let idx = idxs
        .clone()
        .unsqueeze_dim::<2>(1)
        .expand(Shape::new([n, f])); // [N] -> [N, 1] -> [N, F]

    // Reduce the indexed in index matrix, assigned starting at 0, via the add operation.
    // Here, NG is the reduced dim, while N is the original dim. F stays the same.
    let sum = Tensor::zeros(Shape::new([num_graphs, f]), device).scatter(
        0,
        idx.clone(), // [N, F]
        x.clone(),   // [N, F]
        burn::tensor::IndexingUpdateOp::Add,
    ); // [NG, F]

    // Count the nodes per graph so we can produce a [NG, 1] matrix to divide count.
    // This will produce mean.
    let ones = Tensor::<B, 2>::ones(Shape::new([n, 1]), device); // [N, 1]
    let count = Tensor::zeros(Shape::new([num_graphs, 1]), device).scatter(
        0,
        idxs.unsqueeze_dim(1), // [N, 1]
        ones,                  // [N, 1]
        burn::tensor::IndexingUpdateOp::Add,
    ); // [NG, 1] -> [NG, 1]

    // Compute mean.
    sum / count
}
