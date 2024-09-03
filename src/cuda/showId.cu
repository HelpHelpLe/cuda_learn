#include "cuda/showId.cuh"

#include <stdio.h>

__global__ void what_is_my_id(unsigned int *const block,
                              unsigned int *const thread,
                              unsigned int *const warp,
                              unsigned int *const calc_thread) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;

    warp[thread_idx] = threadIdx.x / warpSize;

    calc_thread[thread_idx] = thread_idx;
}

__global__ void
what_is_my_id2(unsigned int *const block_x, unsigned int *const block_y,
               unsigned int *const thread, unsigned int *const calc_thread,
               unsigned int *const x_thread, unsigned int *const y_thread,
               unsigned int *const grid_dimx, unsigned int *const block_dimx,
               unsigned int *const grid_dimy, unsigned int *const block_dimy) {
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

    block_x[thread_idx] = blockIdx.x;
    block_y[thread_idx] = blockIdx.y;
    thread[thread_idx] = threadIdx.x;
    calc_thread[thread_idx] = thread_idx;
    x_thread[thread_idx] = idx;
    y_thread[thread_idx] = idy;
    grid_dimx[thread_idx] = gridDim.x;
    block_dimx[thread_idx] = blockDim.x;
    grid_dimy[thread_idx] = gridDim.y;
    block_dimy[thread_idx] = blockDim.y;
}

#define PROCESS_ARRAY(XX)                                                      \
    XX(block);                                                                 \
    XX(thread);                                                                \
    XX(warp);                                                                  \
    XX(calc_thread);

__host__ void show_id(const unsigned int block_num,
                      const unsigned int thread_num) {
/**
 * define cpu array
 */
#define XX(array) unsigned int cpu_##array[ARRAY_SIZE]
    PROCESS_ARRAY(XX)
#undef XX

/**
 * define gpu array
 */
#define XX(array) unsigned int *gpu_##array
    PROCESS_ARRAY(XX)
#undef XX

/**
 * malloc gpu memory
 */
#define XX(array) cudaMalloc((void **)&gpu_##array, ARRAY_SIZE_IN_BYTES)
    PROCESS_ARRAY(XX)
#undef XX

    what_is_my_id<<<block_num, thread_num>>>(gpu_block, gpu_thread, gpu_warp,
                                             gpu_calc_thread);

/**
 * copy gpu data to cpu
 */
#define XX(array)                                                              \
    cudaMemcpy(cpu_##array, gpu_##array, ARRAY_SIZE_IN_BYTES,                  \
               cudaMemcpyDeviceToHost)
    PROCESS_ARRAY(XX)
#undef XX

/**
 * free gpu memory
 */
#define XX(array) cudaFree(gpu_##array)
    PROCESS_ARRAY(XX)
#undef XX

    for (unsigned int i = 0; i < ARRAY_SIZE; ++i) {
        printf(
            "Calculated Thread: %3u - Block: %2u - Warp: %2u - Thread: %2u\n",
            cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }
}
#undef PROCESS_ARRAY

#define PROCESS_ARRAY(XX)                                                      \
    XX(block_x);                                                               \
    XX(block_y);                                                               \
    XX(thread);                                                                \
    XX(warp);                                                                  \
    XX(calc_thread);                                                           \
    XX(xthread);                                                               \
    XX(ythread);                                                               \
    XX(grid_dimx);                                                             \
    XX(block_dimx);                                                            \
    XX(grid_dimy);                                                             \
    XX(block_dimy);

__host__ void show_id2(const unsigned int threads_x,
                       const unsigned int threads_y,
                       const unsigned int blocks_x,
                       const unsigned int blocks_y) {
    const dim3 threads_rect(threads_x, threads_y);
    const dim3 blocks_rect(blocks_x, blocks_y);
/**
 * define cpu array
 */
#define XX(array) unsigned int cpu_##array[ARRAY_SIZE_Y][ARRAY_SIZE_X]
    PROCESS_ARRAY(XX)
#undef XX

/**
 * define gpu array
 */
#define XX(array) unsigned int *gpu_##array
    PROCESS_ARRAY(XX)
#undef XX

/**
 * malloc gpu memory
 */
#define XX(array) cudaMalloc((void **)&gpu_##array, ARRAY_SIZE_IN_BYTES)
    PROCESS_ARRAY(XX)
#undef XX

    what_is_my_id2<<<blocks_rect, threads_rect>>>(
        gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread, gpu_xthread,
        gpu_ythread, gpu_grid_dimx, gpu_block_dimx, gpu_grid_dimy,
        gpu_block_dimy);

/**
 * copy gpu data to cpu
 */
#define XX(array)                                                              \
    cudaMemcpy(cpu_##array, gpu_##array, ARRAY_SIZE_IN_BYTES,                  \
               cudaMemcpyDeviceToHost)
    PROCESS_ARRAY(XX)
#undef XX

/**
 * free gpu memory
 */
#define XX(array) cudaFree(gpu_##array)
    PROCESS_ARRAY(XX)
#undef XX

    for (int y = 0; y < ARRAY_SIZE_Y; ++y) {
        for (int x = 0; x < ARRAY_SIZE_X; ++x) {
            printf("CT: %2u BKX: %1u BKY: %1u TID: %2u YTID: %2u XTID: %2u "
                   "GDX: %1u BDX: %1u GDY: %1u BDY: %1u\n",
                   cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x],
                   cpu_thread[y][x], cpu_ythread[y][x], cpu_xthread[y][x],
                   cpu_grid_dimx[y][x], cpu_block_dimx[y][x],
                   cpu_grid_dimy[y][x], cpu_block_dimy[y][x]);
        }
    }
}
#undef PROCESS_ARRAY