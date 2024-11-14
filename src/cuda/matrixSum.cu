#include "cuda/matrixSum.cuh"

__global__ void matrix_sum(float *mat_a, float *mat_b, float *mat_c, int nx,
                           int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * nx;
    if (ix < nx && iy < ny) {
        mat_c[idx] = mat_a[idx] + mat_b[idx];
    }
}

__host__ void sum(float *mat_a, float *mat_b, float *mat_c, int nx, int ny) {}