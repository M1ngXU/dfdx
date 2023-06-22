use crate::prelude::{Const, Dtype, Shape, Tape, Tensor};

use super::Device;

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

pub trait FromSparse<
	E: Dtype,
	T: Tape<E, Self>,
	OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>,
>: Device<E>
{
	fn from_sparse(
		&self,
		values: Tensor<(usize,), E, Self, T>,
		indeces: Tensor<(usize, Const<{ OutputShape::NUM_DIMS }>), usize, Self>,
		output_shape: OutputShape,
	) -> Tensor<OutputShape, E, Self, T>;
}

#[cfg(test)]
mod tests {
	use crate::prelude::*;

	#[test]
	fn from_sparse() {
		let dev = AutoDevice::default();

		let indeces = dev
			.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
			.reshape_like(&(5, Const::<2>));
		let values = dev.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).reshape_like(&(5,));

		let (output, mut tape) = dev
			.from_sparse(values.clone().leaky_traced(), indeces, (5, 5))
			.split_tape();

		let expected = vec![
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,
		];

		assert_eq!(output.as_vec(), expected);

		#[cfg(feature = "cuda")]
		{
			use cudarc::CudaSlice;
			dev.dev
				.htod_copy_into(
					(0..25).map(|g| g as f32).collect_vec(),
					&mut **tape.gradients.get_or_alloc_mut(&output).unwrap(),
				)
				.unwrap();
		}
		#[cfg(not(feature = "cuda"))]
		for (i, g) in tape
			.gradients
			.get_or_alloc_mut(&output)
			.unwrap()
			.iter_mut()
			.enumerate()
		{
			*g = i as f32;
		}

		let grads = tape.execute().unwrap();

		let gradients = grads.get(&values);
		let expected = vec![0.0, 6.0, 12.0, 18.0, 24.0];
		assert_eq!(gradients.as_vec(), expected);
	}
}
