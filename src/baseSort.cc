#include "baseSort.h"

#include <vector>

void cpu_sort(uint32_t * const data,
              const uint32_t num_elements) {
    std::vector<uint32_t> cpu_tmp_0(num_elements);
    std::vector<uint32_t> cpu_tmp_1(num_elements);

    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t base_cnt_0 = 0;
        uint32_t base_cnt_1 = 0;

        for (uint32_t i = 0; i < num_elements; ++i) {
            const uint32_t d = data[i];
            const uint32_t bit_mask = (1 << bit);

            if ((d & bit_mask) > 0) {
                cpu_tmp_1[base_cnt_1] = d;
                ++base_cnt_1;
            } else {
                cpu_tmp_0[base_cnt_0] = d;
                ++base_cnt_0;
            }
        }

        for (uint32_t i = 0; i < base_cnt_0; ++i) {
            data[i] = cpu_tmp_0[i];
        }

        for (uint32_t i = 0; i < base_cnt_1; ++i) {
            data[base_cnt_0 + i] = cpu_tmp_1[i];
        }
    }
}