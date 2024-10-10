#include "cuda/baseSort.cuh"

__device__ void radix_sort(uint32_t * const sort_tmp,
                           const uint32_t num_lists,
                           const uint32_t num_elements,
                           const uint32_t tid,
                           uint32_t * const sort_tmp_1) {
    
    for (uint32_t bit = 0; bit < 32; ++bit) {
        const uint32_t bit_mask = (1 << bit);
        uint32_t base_cnt_0 = 0;
        uint32_t base_cnt_1 = 0;

        for (uint32_t i = 0; i < num_elements; i += num_lists) {
            const uint32_t elem = sort_tmp[i + tid];

            if ((elem & bit_mask) > 0) {
                sort_tmp_1[base_cnt_1 + tid] = elem;
                base_cnt_1 += num_lists;
            } else {
                sort_tmp[base_cnt_0 + tid] = elem;
                base_cnt_0 += num_lists;
            }
        }

        for (uint32_t i = 0; i < base_cnt_1; i += num_lists) {
            sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
        }
    }

    __syncthreads();
}

__device__ void copy_data_to_shared(const uint32_t * const data,
                                    uint32_t * const sort_tmp,
                                    const uint32_t num_lists,
                                    const uint32_t num_elements,
                                    const uint32_t tid) {
    for (uint32_t i = 0; i < num_elements; i += num_lists) {
        sort_tmp[i + tid] = data[i + tid];
    }

    __syncthreads();
}

__device__ void merge_array(const uint32_t * const src_array,
                            uint32_t * const des_array,
                            const uint32_t num_lists,
                            const uint32_t num_elements,
                            const uint32_t tid) {
    __shared__ uint32_t list_indexes[MAX_NUM_LISTS];

    list_indexes[tid] = 0;
    __syncthreads();

    if (tid == 0) {
        const uint32_t num_elements_per_list = (num_elements / num_lists);

        for (uint32_t i = 0; i < num_elements; ++i) {
            uint32_t min_val = 0xFFFFFFFF;
            uint32_t min_idx = 0;

            for (uint32_t list = 0; list < num_lists; ++list) {
                if (list_indexes[list] < num_elements_per_list) {
                    const uint32_t src_idx = list + (list_indexes[list] * num_lists);

                    const uint32_t data = src_array[src_idx];
                    if (data <= min_val) {
                        min_val = data;
                        min_idx = list;
                    }
                }
            }
            ++list_indexes[min_idx];
            des_array[i] = min_val;
        }
    }
}

__global__ void gpu_sort_array_array(uint32_t * const data,
                                     const uint32_t num_lists,
                                     const uint32_t num_elements) {
    // const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t sort_tmp[NUM_ELEM];
    __shared__ uint32_t sort_tmp_1[NUM_ELEM];

    copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);

    radix_sort(sort_tmp, num_lists, num_elements, tid, sort_tmp_1);

    merge_array(sort_tmp, data, num_lists, num_elements, tid);
}

__host__ void gpu_sort(uint32_t * const data,
                       const uint32_t num_elements) {
    uint32_t * gpu_data;

    cudaMalloc((void **)&gpu_data, num_elements * sizeof(uint32_t));

    cudaMemcpy(gpu_data, data, num_elements, cudaMemcpyHostToDevice);

    gpu_sort_array_array<<<1, MAX_NUM_LISTS>>>(gpu_data, MAX_NUM_LISTS, num_elements);
    
    cudaMemcpy(data, gpu_data, num_elements, cudaMemcpyDeviceToHost);

    cudaFree(gpu_data);
}