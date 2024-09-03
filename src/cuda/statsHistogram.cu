#include "cuda/statsHistogram.cuh"

__shared__ unsigned int d_bin_data_shared[256];

#define ATOMIC_ADD_TO_BIN_SHARED(value, shift)                                 \
    atomicAdd(&(d_bin_data_shared[((value) >> (shift)) & 0xFF]), 1)

__global__ void histogram_kernel(const unsigned int *const d_hist_data,
                                 unsigned int *const d_bin_data) {
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = gridDim.x * blockDim.x * idy + idx;

    d_bin_data_shared[threadIdx.x] = 0;

    const unsigned int value_u32 = d_hist_data[tid];

    __syncthreads();

    ATOMIC_ADD_TO_BIN_SHARED(value_u32, 0);
    ATOMIC_ADD_TO_BIN_SHARED(value_u32, 8);
    ATOMIC_ADD_TO_BIN_SHARED(value_u32, 16);
    ATOMIC_ADD_TO_BIN_SHARED(value_u32, 24);

    __syncthreads();

    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}

__host__ void stats_histogram(const unsigned char *const hist_data,
                              const unsigned int hist_size,
                              unsigned int *const bin_data) {
    unsigned int *d_hist_data;
    unsigned int *d_bin_data;

    cudaMalloc((void **)&d_hist_data, hist_size);
    cudaMalloc((void **)&d_bin_data, 256 * sizeof(unsigned int));

    cudaMemcpy(d_hist_data, hist_data, hist_size, cudaMemcpyHostToDevice);

    histogram_kernel<<<1, 256>>>(d_hist_data, d_bin_data);

    cudaMemcpy(bin_data, d_bin_data, 256 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_hist_data);
    cudaFree(d_bin_data);
}