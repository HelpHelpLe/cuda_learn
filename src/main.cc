#include "cuda/matrixMul.cuh"
#include "cuda/showId.cuh"

#include <stdio.h>

int main() {
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

    // show_id(2, 64);

    // printf("Total thread count = 32 * 4 = 128\n");
    // show_id2(32, 4, 1, 4);
    // printf("Total thread count = 16 * 8 = 128\n");
    // show_id2(16, 8, 2, 2);
    return 0;
}