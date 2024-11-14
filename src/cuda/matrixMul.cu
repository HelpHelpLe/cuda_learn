#include "cuda/matrixMul.cuh"

__device__ int sum(int *cache, int id) {
    int i = blockDim.x / 2;
    while (i != 0) {
        if (id < i) {
            cache[id] += cache[id + i];
        }
        __syncthreads();
        i /= 2;
    }
    return cache[0];
}

__global__ void matrix_dot(int *a, int *b, int *c) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    __shared__ int cache[BLOCK];
    int t = threadIdx.x;
    if (t < N)
        cache[t] = a[i * N + t] * b[t * N + j];
    else
        cache[t] = 0;
    __syncthreads();
    sum(cache, t);
    __syncthreads();
    c[i * N + j] = cache[0];
}

#define PROCESS_ARRAY(XX) \
    XX(a);                \
    XX(b);                \
    XX(c);

__host__ void dot(int *a, int *b, int *c) {
    int *a_cuda, *b_cuda, *c_cuda;

#define XX(array) cudaMalloc((void **)&array##_cuda, MATRIX_SIZE_IN_BYTES)
    PROCESS_ARRAY(XX)
#undef XX

    cudaMemcpy(a_cuda, a, MATRIX_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, MATRIX_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
    dim3 matrix(N, N);
    matrix_dot<<<matrix, BLOCK>>>(a_cuda, b_cuda, c_cuda);
    cudaMemcpy(c, c_cuda, MATRIX_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

#define XX(array) cudaFree(array##_cuda)
    PROCESS_ARRAY(XX)
#undef XX
}
#undef PROCESS_ARRAY