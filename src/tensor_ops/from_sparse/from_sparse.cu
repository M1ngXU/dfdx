#include "cuda_utils.cuh"

template<typename T>
__device__ void from_sparse_fwd(
    const size_t numel,
    const T * values,
    const size_t *values_info,
    const size_t *indeces,
    const size_t *indeces_info,
    T* output,
    const size_t *output_info,
    const size_t num_dims
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const size_t *values_dims = values_info;
    const size_t *values_strides = values_info + 1;
    const size_t *indeces_dims = indeces_info;
    const size_t *indeces_strides = indeces_info + 2;
    const size_t *output_dims = output_info;
    const size_t *output_strides = output_info + num_dims;

    unsigned int index = 0;
    for (unsigned int d = num_dims - 1; d < num_dims; d--) {
        index += get_strided_index(i * num_dims + d, 2, indeces_dims, indeces_strides) * output_dims[d];
    }
    float value = values[get_strided_index(i, 1, values_dims, values_strides)];
    output[get_strided_index(index, num_dims, output_dims, output_strides)] = value;
}

template<typename T>
__device__ void from_sparse_bwd(
    const size_t numel,
    T * values,
    const size_t *values_info,
    const size_t *indeces,
    const size_t *indeces_info,
    const T* output,
    const size_t *output_info,
    const size_t num_dims
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const size_t *values_dims = values_info;
    const size_t *values_strides = values_info + 1;
    const size_t *indeces_dims = indeces_info;
    const size_t *indeces_strides = indeces_info + 2;
    const size_t *output_dims = output_info;
    const size_t *output_strides = output_info + num_dims;

    unsigned int index = 0;
    for (unsigned int d = num_dims - 1; d < num_dims; d--) {
        index += get_strided_index(i * num_dims + d, 2, indeces_dims, indeces_strides) * output_dims[d];
    }
    float value = output[get_strided_index(index, num_dims, output_dims, output_strides)];
    values[get_strided_index(i, 1, values_dims, values_strides)] = value;
}

#define FROM_SPARSE(TYPENAME, FWD, BWD) \
extern "C" __global__ void FWD( \
    const size_t numel, \
    const TYPENAME * values, \
    const size_t *values_info, \
    const size_t *indeces, \
    const size_t *indeces_info, \
    TYPENAME* output, \
    const size_t *output_info, \
    const size_t num_dims \
) { \
    from_sparse_fwd(numel, values, values_info, indeces, indeces_info, output, output_info, num_dims); \
} \
extern "C" __global__ void BWD( \
    const size_t numel, \
    TYPENAME * values, \
    const size_t *values_info, \
    const size_t *indeces, \
    const size_t *indeces_info, \
    const TYPENAME* output, \
    const size_t *output_info, \
    const size_t num_dims \
) { \
    from_sparse_bwd(numel, values, values_info, indeces, indeces_info, output, output_info, num_dims); \
}

FROM_SPARSE(__half, from_sparse_fwd_f16, from_sparse_bwd_f16);
FROM_SPARSE(float, from_sparse_fwd_f32, from_sparse_bwd_f32);
FROM_SPARSE(double, from_sparse_fwd_f64, from_sparse_bwd_f64);