#pragma once

#define N 10
#define BLOCK 16

constexpr unsigned int MATRIX_SIZE_IN_BYTES = N * N * sizeof(int);

void dot(int *a, int *b, int *c);