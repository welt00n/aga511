#ifndef GPU_DEVICE_CUH
#define GPU_DEVICE_CUH

#include <cuda_runtime.h>

#include "run/run.cuh"

void check(cudaError_t error, const char *what);
cudaDeviceProp gpu_properties(void);
int gpu_attribute(cudaDeviceAttr attribute);
void describe_gpu(ProcessorInfo *info);

#endif
