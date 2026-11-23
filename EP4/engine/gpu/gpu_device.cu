#include <stdio.h>
#include <stdlib.h>

#include "gpu/gpu_device.cuh"

#define MEMORY_HEADROOM_BYTES (256ULL << 20)

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

static double peak_bandwidth(void) {
    double memory_clock_hz = gpu_attribute(cudaDevAttrMemoryClockRate) * 1e3;
    double bus_bytes = gpu_attribute(cudaDevAttrGlobalMemoryBusWidth) / 8.0;
    return 2.0 * memory_clock_hz * bus_bytes;
}

static ProcessorHardware describe_gpu_hardware(void) {
    ProcessorHardware hardware;
    check(cudaMemGetInfo(&hardware.memory_free, &hardware.memory_total), "reading the GPU's free memory");
    hardware.memory_for_image = hardware.memory_free > MEMORY_HEADROOM_BYTES ?
                                hardware.memory_free - MEMORY_HEADROOM_BYTES : 0;
    hardware.l2_bytes = (size_t) gpu_attribute(cudaDevAttrL2CacheSize);
    hardware.peak_bandwidth = peak_bandwidth();
    hardware.fp32_per_fp64 = gpu_attribute(cudaDevAttrSingleToDoublePrecisionPerfRatio);
    hardware.max_threads_per_unit = gpu_attribute(cudaDevAttrMaxThreadsPerMultiProcessor);
    check(cudaRuntimeGetVersion(&hardware.runtime_version), "reading the CUDA runtime version");
    check(cudaDriverGetVersion(&hardware.driver_version), "reading the CUDA driver version");
    return hardware;
}

void describe_gpu(ProcessorInfo *info) {
    cudaDeviceProp gpu = gpu_properties();
    info->kind = "GPU";
    snprintf(info->name, PROCESSOR_NAME_LENGTH, "%s", gpu.name);
    replace_spaces_with_underscores(info->name);
    snprintf(info->compute_capability, COMPUTE_CAPABILITY_LENGTH, "%d.%d", gpu.major, gpu.minor);
    info->compute_units_total = (unsigned int) gpu.multiProcessorCount;
    info->compute_units = info->compute_units_total;
    info->unit = "sm";
    info->hardware = describe_gpu_hardware();
}
