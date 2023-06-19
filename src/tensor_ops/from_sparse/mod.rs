use crate::prelude::{Dtype, Shape, Tape, Tensor};

use super::Device;

mod cpu_kernel;

// TODO TryFromSparse
pub trait FromSparse<E: Dtype, D: Device<E>, T: Tape<E, D>, InputShape: Shape, OutputShape: Shape> {
    fn from_sparse(
        &self,
        sparse: Tensor<InputShape, E, D, T>,
        output_shape: OutputShape,
    ) -> Tensor<OutputShape, E, D, T>;
}
