#pragma once

#define NUM_ELEM 1024
#define MAX_NUM_LISTS 256


void gpu_sort(uint32_t * const data,
              const uint32_t num_elements);