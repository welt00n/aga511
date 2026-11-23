#ifndef GPU_RNG_CUH
#define GPU_RNG_CUH

#include <curand_kernel.h>

#include "physics/real.cuh"

typedef curandStatePhilox4_32_10_t GpuRng;

static inline __device__ real uniform(GpuRng *state) {
#ifdef STAR_FP32
    return curand_uniform(state);
#else
    return curand_uniform_double(state);
#endif
}

#endif
