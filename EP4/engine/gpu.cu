#include <stdio.h>
#include <string.h>

#include "processor.cuh"
#include "gpu_device.cuh"
#include "gpu_rng.cuh"
#include "rng.cuh"

#define NUMBERS_PER_CHUNK_CHECKED 16

static const unsigned long long CHUNKS_CHECKED[] = {0, 1, 1000003, 4294967301ULL};
static const int CHUNK_CHECKS = sizeof(CHUNKS_CHECKED) / sizeof(CHUNKS_CHECKED[0]);

struct Processor {
    const Options *options;
};


// ============================================================ random numbers: cuRAND vs rng.cuh

__global__ void draw_first_numbers(unsigned long long seed, const unsigned long long *chunks,
                                   unsigned int *bits, real *uniforms) {
    unsigned int check = threadIdx.x;
    GpuRng rng;
    curand_init(seed, chunks[check], 0, &rng);
    for (int i = 0; i < NUMBERS_PER_CHUNK_CHECKED; i++) {
        bits[check * NUMBERS_PER_CHUNK_CHECKED + i] = curand(&rng);
    }
    curand_init(seed, chunks[check], 0, &rng);
    for (int i = 0; i < NUMBERS_PER_CHUNK_CHECKED; i++) {
        uniforms[check * NUMBERS_PER_CHUNK_CHECKED + i] = uniform(&rng);
    }
}

static void draw_on_gpu(unsigned long long seed, unsigned int *bits, real *uniforms) {
    size_t numbers = CHUNK_CHECKS * NUMBERS_PER_CHUNK_CHECKED;
    unsigned long long *gpu_chunks;
    unsigned int *gpu_bits;
    real *gpu_uniforms;
    check(cudaMalloc(&gpu_chunks, sizeof(CHUNKS_CHECKED)), "allocating the chunks to check");
    check(cudaMalloc(&gpu_bits, numbers * sizeof(unsigned int)), "allocating the numbers");
    check(cudaMalloc(&gpu_uniforms, numbers * sizeof(real)), "allocating the uniforms");
    check(cudaMemcpy(gpu_chunks, CHUNKS_CHECKED, sizeof(CHUNKS_CHECKED), cudaMemcpyHostToDevice), "copying the chunks");
    draw_first_numbers<<<1, CHUNK_CHECKS>>>(seed, gpu_chunks, gpu_bits, gpu_uniforms);
    check(cudaGetLastError(), "launching draw_first_numbers");
    check(cudaMemcpy(bits, gpu_bits, numbers * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copying the numbers");
    check(cudaMemcpy(uniforms, gpu_uniforms, numbers * sizeof(real), cudaMemcpyDeviceToHost), "copying the uniforms");
    cudaFree(gpu_chunks);
    cudaFree(gpu_bits);
    cudaFree(gpu_uniforms);
}

static bool same_bits_as_cpu(unsigned long long seed, unsigned long long chunk, const unsigned int *gpu_bits) {
    PhiloxState rng;
    philox_init(seed, chunk, 0, &rng);
    for (int i = 0; i < NUMBERS_PER_CHUNK_CHECKED; i++) {
        if (philox_next(&rng) != gpu_bits[i]) {
            return false;
        }
    }
    return true;
}

static bool same_uniforms_as_cpu(unsigned long long seed, unsigned long long chunk, const real *gpu_uniforms) {
    PhiloxState rng;
    philox_init(seed, chunk, 0, &rng);
    for (int i = 0; i < NUMBERS_PER_CHUNK_CHECKED; i++) {
        real cpu_uniform = uniform(&rng);
        if (memcmp(&cpu_uniform, &gpu_uniforms[i], sizeof(real)) != 0) {
            return false;
        }
    }
    return true;
}

int compare_rng_with_cpu(unsigned long long seed) {
    unsigned int bits[CHUNK_CHECKS * NUMBERS_PER_CHUNK_CHECKED];
    real uniforms[CHUNK_CHECKS * NUMBERS_PER_CHUNK_CHECKED];
    draw_on_gpu(seed, bits, uniforms);
    printf("GPU_SEED_%llu_CHUNK_0=", seed);
    for (int i = 0; i < 8; i++) {
        printf("%08x%s", bits[i], i < 7 ? "," : "\n");
    }
    int failures = 0;
    for (int c = 0; c < CHUNK_CHECKS; c++) {
        bool bits_match = same_bits_as_cpu(seed, CHUNKS_CHECKED[c], &bits[c * NUMBERS_PER_CHUNK_CHECKED]);
        bool uniforms_match = same_uniforms_as_cpu(seed, CHUNKS_CHECKED[c], &uniforms[c * NUMBERS_PER_CHUNK_CHECKED]);
        printf("GPU_RNG_CHUNK=%llu BITS=%s UNIFORMS=%s\n", CHUNKS_CHECKED[c],
               bits_match ? "SAME_AS_CPU" : "DIFFERENT", uniforms_match ? "SAME_AS_CPU" : "DIFFERENT");
        failures += (bits_match ? 0 : 1) + (uniforms_match ? 0 : 1);
    }
    return failures;
}


// ============================================================ the processor

void describe_processor(const Options *options, ProcessorInfo *info) {
    describe_gpu(info);
}

Processor *open_processor(const Options *options) {
    fprintf(stderr, "The GPU does not trace photons yet.\n");
    return NULL;
}

void load_pass(Processor *processor, const Scene *scene) {
}

void zero_pass_image(Processor *processor, unsigned int observers) {
}

PhotonCounts trace_pass(Processor *processor, const Run *run, Progress *progress) {
    PhotonCounts counts = {0, 0, 0, 0};
    return counts;
}

const PixelCount *observer_on_host(Processor *processor, unsigned int slot) {
    return NULL;
}

size_t processor_image_bytes(const Processor *processor) {
    return 0;
}

void close_processor(Processor *processor) {
}
