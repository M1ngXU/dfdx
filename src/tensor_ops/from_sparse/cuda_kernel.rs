use crate::prelude::*;
use cudarc::driver::{CudaSlice, LaunchAsync};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/from_sparse.ptx"));

macro_rules! impl_cuda_kernel {
	($type:ty, $mod:tt, $fwd:tt, $bwd:tt $(,)?) => {
		impl<
				T: Tape<$type, Self> + 'static,
				OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>,
			> FromSparse<$type, T, OutputShape> for Cuda
		{
			fn from_sparse(
				&self,
				values: Tensor<(usize,), $type, Self, T>,
				indeces: Tensor<(usize, Const<{ OutputShape::NUM_DIMS }>), usize, Self>,
				output_shape: OutputShape,
			) -> Tensor<OutputShape, $type, Self, T> {
				assert_eq!(values.shape().0, indeces.shape().0);

				if !self.dev.has_func($mod, $fwd) {
					self.dev
						.load_ptx(PTX_SRC.into(), $mod, &[$fwd, $bwd])
						.unwrap();
				}

				let fwd_fn = self.dev.get_func($mod, $fwd).unwrap();
				let bwd_fn = self.dev.get_func($mod, $bwd).unwrap();

				let (values, mut tape) = values.split_tape();

				let cfg = launch_cfg::<128>(values.shape().0 as u32);

				let mut values_info = Vec::with_capacity(1 * 2);
				values_info.push(values.shape().concrete()[0]);
				values_info.push(values.strides()[0]);
				let values_info = self.dev.htod_copy(values_info).unwrap();

				let mut indeces_info = Vec::with_capacity(2 * 2);
				indeces_info.extend(indeces.shape().concrete());
				indeces_info.extend(indeces.strides());
				let indeces_info = self.dev.htod_copy(indeces_info).unwrap();

				let mut output = self.dev.alloc_zeros(output_shape.num_elements()).unwrap();
				let mut output_info = Vec::with_capacity(OutputShape::NUM_DIMS * 2);
				output_info.extend(output_shape.concrete());
				output_info.extend(output_shape.strides());
				let output_info = self.dev.htod_copy(output_info).unwrap();

				let params: (
					usize,
					&CudaSlice<$type>,
					&CudaSlice<usize>,
					&CudaSlice<usize>,
					&CudaSlice<usize>,
					&mut CudaSlice<$type>,
					&CudaSlice<usize>,
					usize,
				) = (
					values.shape().0,
					&**values.data,
					&values_info,
					&**indeces.data,
					&indeces_info,
					&mut output,
					&output_info,
					OutputShape::NUM_DIMS,
				);
				unsafe {
					fwd_fn.launch(cfg, params).unwrap();
				}

				let inp_ghost = values.clone();
				let output = self.build_tensor(output_shape, output_shape.strides(), output);
				let out_ghost = output.clone();
				tape.add_backward_op(move |grads| {
					grads.try_alloc_for(&inp_ghost)?;
					grads.try_alloc_for(&out_ghost)?;
					let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);

					let params: (
						usize,
						&mut CudaSlice<$type>,
						&CudaSlice<usize>,
						&CudaSlice<usize>,
						&CudaSlice<usize>,
						&CudaSlice<$type>,
						&CudaSlice<usize>,
						usize,
					) = (
						values.shape().0,
						&mut **grad_inp,
						&values_info,
						&**indeces.data().unwrap(),
						&indeces_info,
						&**grad_out,
						&output_info,
						OutputShape::NUM_DIMS,
					);
					unsafe {
						bwd_fn.launch(cfg, params).unwrap();
					}
					Ok(())
				});
				output.put_tape(tape)
			}
		}
	};
}

#[cfg(feature = "f16")]
impl_cuda_kernel!(
	half::f16,
	"from_sparse_f16",
	"from_sparse_fwd_f16",
	"from_sparse_bwd_f16",
);
impl_cuda_kernel!(
	f32,
	"from_sparse_f32",
	"from_sparse_fwd_f32",
	"from_sparse_bwd_f32",
);
impl_cuda_kernel!(
	f64,
	"from_sparse_f64",
	"from_sparse_fwd_f64",
	"from_sparse_bwd_f64",
);
