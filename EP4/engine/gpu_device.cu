#include <stdio.h>
#include <stdlib.h>

#include "gpu_device.cuh"

void check(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error while %s: %s\n", what, cudaGetErrorString(error));
        exit(1);
    }
}

cudaDeviceProp gpu_properties(void) {
    cudaDeviceProp gpu;
    check(cudaGetDeviceProperties(&gpu, 0), "reading the GPU properties");
    return gpu;
}

int gpu_attribute(cudaDeviceAttr attribute) {
    int value = 0;
    check(cudaDeviceGetAttribute(&value, attribute, 0), "reading a GPU attribute");
    return value;
}

void describe_gpu(ProcessorInfo *info) {
    cudaDeviceProp gpu = gpu_properties();
    info->kind = "GPU";
    snprintf(info->name, PROCESSOR_NAME_LENGTH, "%s", gpu.name);
    replace_spaces_with_underscores(info->name);
    snprintf(info->compute_capability, COMPUTE_CAPABILITY_LENGTH, "%d.%d", gpu.major, gpu.minor);
    info->compute_units = (unsigned int) gpu.multiProcessorCount;
    info->unit = "sm";
}
