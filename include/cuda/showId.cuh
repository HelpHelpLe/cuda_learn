#pragma once

constexpr unsigned int ARRAY_SIZE = 128;
constexpr unsigned int ARRAY_SIZE_X = 32;
constexpr unsigned int ARRAY_SIZE_Y = 16;
constexpr unsigned int ARRAY_SIZE_IN_BYTES =
    (sizeof(unsigned int) * ARRAY_SIZE);

void show_id(const unsigned int block_num, const unsigned int thread_num);

void show_id2(const unsigned int threads_x, const unsigned int threads_y,
              const unsigned int blocks_x, const unsigned int blocks_y);