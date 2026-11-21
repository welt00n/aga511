#ifndef GPU_IMAGE_CUH
#define GPU_IMAGE_CUH

#include <stddef.h>

#include "image.cuh"

typedef struct {
    PixelCount *counts;
    size_t pixels_per_observer;
    unsigned int observers;
} GpuImage;

void allocate_gpu_image(GpuImage *image, size_t pixels_per_observer, unsigned int observers);
void zero_gpu_image(GpuImage *image, unsigned int observers);
void copy_observer_to_host(const GpuImage *image, unsigned int slot, PixelCount *host_counts);
void free_gpu_image(GpuImage *image);
size_t gpu_image_bytes(const GpuImage *image);

static inline __device__ void count_photon_atomic(PixelCount *counts, unsigned long long pixel) {
    atomicAdd(&counts[pixel], 1u);
}

#endif
