#include <stdio.h>

#include "cuda/sample.cuh"

void test_block() {
    int n_elem = 1024;
    dim3 block(1024);
    dim3 grid((n_elem - 1) / block.x + 1);
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);
    printf("block.x: %d, block.y: %d, block.z: %d\n", block.x, block.y,
           block.z);
}

void check_device() {
    int dev_cnt = 0;
    cudaError_t error_id = cudaGetDeviceCount(&dev_cnt);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n ->%s\n", (int)error_id,
               cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (dev_cnt == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", dev_cnt);
    }

    int dev = 0, driver_version = 0, runtime_version = 0;

    cudaSetDevice(dev);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("Device %d:\"%s\"\n", dev, device_prop.name);
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf("  CUDA Driver Version / Runtime Version         %d.%d / %d.%d\n",
           driver_version / 1000, (driver_version % 100) / 10,
           runtime_version / 1000, (runtime_version % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:   %d.%d\n",
           device_prop.major, device_prop.minor);
    printf(
        "  Total amount of global memory:                %.2f MBytes (%llu "
        "bytes)\n",
        (float)device_prop.totalGlobalMem / pow(1024.0, 3));
    printf(
        "  GPU Clock rate:                               %.0f MHz (%0.2f "
        "GHz)\n",
        device_prop.clockRate * 1e-3f, device_prop.clockRate * 1e-6f);
    printf("  Memory Bus width:                             %d-bits\n",
           device_prop.memoryBusWidth);

    if (device_prop.l2CacheSize) {
        printf("  L2 Cache Size:                            	%d bytes\n",
               device_prop.l2CacheSize);
    }
    printf(
        "  Max Texture Dimension Size (x,y,z)            "
        "1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
        device_prop.maxTexture1D, device_prop.maxTexture2D[0],
        device_prop.maxTexture2D[1], device_prop.maxTexture3D[0],
        device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
    printf(
        "  Max Layered Texture Size (dim) x layers       1D=(%d) x "
        "%d,2D=(%d,%d) x %d\n",
        device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1],
        device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1],
        device_prop.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory               %lu bytes\n",
           device_prop.totalConstMem);
    printf("  Total amount of shared memory per block:      %lu bytes\n",
           device_prop.sharedMemPerBlock);
    printf("  Total number of registers available per block:%d\n",
           device_prop.regsPerBlock);
    printf("  Wrap size:                                    %d\n",
           device_prop.warpSize);
    printf("  Maximun number of thread per multiprocesser:  %d\n",
           device_prop.maxThreadsPerMultiProcessor);
    printf("  Maximun number of thread per block:           %d\n",
           device_prop.maxThreadsPerBlock);
    printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
           device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
           device_prop.maxThreadsDim[2]);
    printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
           device_prop.maxGridSize[0], device_prop.maxGridSize[1],
           device_prop.maxGridSize[2]);
    printf("  Maximu memory pitch                           %lu bytes\n",
           device_prop.memPitch);
    exit(EXIT_SUCCESS);
}