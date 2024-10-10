#include "cuda/matrixMul.cuh"
#include "cuda/showId.cuh"
#include "cuda/statsHistogram.cuh"
#include "baseSort.h"
#include "cuda/baseSort.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void test_dot() {
    int a[N * N], b[N * N], c[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i;
            b[i * N + j] = j;
        }
    }
    dot(a, b, c);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", c[i * N + j]);
        }
        printf("\n");
    }
}

void test_show_id() { show_id(2, 64); }

void test_show_id2() {
    printf("Total thread count = 32 * 4 = 128\n");
    show_id2(32, 4, 1, 4);
    printf("Total thread count = 16 * 8 = 128\n");
    show_id2(16, 8, 2, 2);
}

void test_stats_histogram() {
    constexpr unsigned int n = 1024;
    unsigned char hist_data[n];
    unsigned int bin_data[256];
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++) {
        hist_data[i] = (unsigned char)(rand() % 256);
    }

    stats_histogram(hist_data, n, bin_data);

    unsigned int sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += bin_data[i];
        printf("The num of %3d is: %3u\n", i, bin_data[i]);
    }
    printf("The total num is: %u\n", sum);
    assert(sum == n);
}

void test_gpu_base_sort() {
    uint32_t arr[NUM_ELEM];

    srand(static_cast<unsigned>(time(nullptr)));

    for (int i = 0; i < NUM_ELEM; ++i) {
        arr[i] = rand() % 10000;
    }

    printf("Random array: ");
    for (int i = 0; i < NUM_ELEM; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    gpu_sort(arr, NUM_ELEM);

    printf("Sorted array: ");
    for (int i = 0; i < NUM_ELEM; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    test_gpu_base_sort();
    
    return 0;
}