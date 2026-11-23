#include "gpu/gpu_image.cuh"
#include "gpu/gpu_device.cuh"

static size_t observer_bytes(const GpuImage *image) {
    return image->pixels_per_observer * sizeof(PixelCount);
}

void allocate_gpu_image(GpuImage *image, size_t pixels_per_observer, unsigned int observers) {
    image->pixels_per_observer = pixels_per_observer;
    image->observers = observers;
    check(cudaMalloc(&image->counts, gpu_image_bytes(image)), "allocating the image on the GPU");
}

void zero_gpu_image(GpuImage *image, unsigned int observers) {
    check(cudaMemset(image->counts, 0, (size_t) observers * observer_bytes(image)), "zeroing the image on the GPU");
}

void copy_observer_to_host(const GpuImage *image, unsigned int slot, PixelCount *host_counts) {
    const PixelCount *block = image->counts + (size_t) slot * image->pixels_per_observer;
    check(cudaMemcpy(host_counts, block, observer_bytes(image), cudaMemcpyDeviceToHost), "copying an observer back");
}

void free_gpu_image(GpuImage *image) {
    cudaFree(image->counts);
    image->counts = NULL;
}

size_t gpu_image_bytes(const GpuImage *image) {
    return (size_t) image->observers * observer_bytes(image);
}
