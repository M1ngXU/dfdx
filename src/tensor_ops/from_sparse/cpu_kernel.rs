use crate::prelude::{
    cpu::{index_to_i, CachableVec},
    Const, Cpu, Device, Dtype, Gradients, PutTape, Rank1, Rank2, ReshapeTo, Shape, SplitTape,
    Storage, Tape, Tensor, ZerosTensor,
};

use super::FromSparse;

impl<
        E: Dtype,
        T: Tape<E, Cpu>,
        // The requirement for `Concrete` to be `[usize; ...]` is required to "chunk"
        // the `indeces` flattened tensor into a `Vec<[usize; ...]>`
        //
        // this might not be possible for a custom `Concrete` type (so that specific one is required)
        OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>,
    > FromSparse<E, Cpu, T, (usize,), OutputShape>
    for Tensor<(usize, Const<{ OutputShape::NUM_DIMS }>), usize, Cpu>
where
    Cpu: Device<E>,
{
    fn from_sparse(
        &self,
        sparse: Tensor<(usize,), E, Cpu, T>,
        output_shape: OutputShape,
    ) -> Tensor<OutputShape, E, Cpu, T> {
        assert_eq!(
            sparse.shape.0, self.shape.0,
            "There need to be the same amount of indeces as the amount of values to create a tensor from a sparse one."
        );
        let (values, mut tape) = sparse.split_tape();
        let mut data: Tensor<OutputShape, E, Cpu> = Cpu::default().zeros_like(&output_shape);
        let indeces = self
            .as_vec()
            // use `.array_chunks()` if stabilized
            .chunks(OutputShape::NUM_DIMS)
            .map(|s| *unsafe { &*(s.as_ptr() as *const [usize; OutputShape::NUM_DIMS]) })
            .collect::<Vec<_>>();
        for (index, value) in indeces.iter().zip(values.as_vec()) {
            data[*index] = value;
        }
        let inp_ghost = values.ghost();
        let out_ghost = data.ghost();

        let input_shape = values.shape;
        let input_strides = values.strides;
        tape.add_backward_op(move |grads: &mut Gradients<E, Cpu>| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            let grad_inp = unsafe { &mut *(grad_inp as *mut _ as *mut CachableVec<E>) };
            let grad_out = unsafe { &*(grad_out as *const _ as *const CachableVec<E>) };
            for (i, index) in indeces.into_iter().enumerate() {
                grad_inp[index_to_i(&input_shape, &input_strides, [i])] +=
                    grad_out[index_to_i(&output_shape, &output_shape.strides(), index)];
            }
            Ok(())
        });
        data.put_tape(tape)
    }
}
// impl<
//         const L: usize,
//         E: Dtype,
//         T: Tape<E, Cpu>,
//         OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>,
//     > FromSparse<E, Cpu, T, Rank1<L>, OutputShape>
//     for Tensor<Rank2<L, { OutputShape::NUM_DIMS }>, usize, Cpu>
// where
//     Cpu: Device<E>,
// {
//     fn from_sparse(
//         &self,
//         sparse: Tensor<Rank1<L>, E, Cpu, T>,
//         output_shape: OutputShape,
//     ) -> Tensor<OutputShape, E, Cpu, T> {
//         self.clone()
//             .reshape_like(&(L, Const::<{ OutputShape::NUM_DIMS }>))
//             .from_sparse(sparse.reshape_like(&(L,)), output_shape)
//     }
// }

#[cfg(test)]
mod tests {
    use crate::{data::OneHotEncode, prelude::*};

    #[test]
    fn test_from_sparse() {
        let dev = Cpu::default();
        let gradients: Gradients<f32, Cpu> = Gradients::leaky();
        let sparse: Tensor<(usize,), f32, Cpu> = dev.ones_like(&(5,));
        let indeces: Tensor<(usize, Const<2>), usize, Cpu> =
            dev.tensor_from_vec(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4], (5, Const::<2>));
        let (output, mut tape): (Tensor<(usize, usize), _, _>, _) = indeces
            .from_sparse(sparse.clone().traced(gradients), (5, 5))
            .split_tape();
        for (i, e) in tape
            .gradients
            .get_or_alloc_mut(&output)
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *e = i as f32;
        }
        assert_eq!(
            output.as_vec(),
            dev.one_hot_encode(Const::<5>, [0, 1, 2, 3, 4]).as_vec()
        );
        let gradients = tape.execute().unwrap();
        assert_eq!(
            gradients.get(&sparse).as_vec(),
            vec![0.0, 6.0, 12.0, 18.0, 24.0]
        );
    }
}
