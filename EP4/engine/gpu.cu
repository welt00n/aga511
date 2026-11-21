#include <stdio.h>
#include <stdlib.h>

#include "processor.cuh"
#include "photon.cuh"
#include "gpu_device.cuh"
#include "gpu_image.cuh"
#include "gpu_rng.cuh"
#include "gpu_sms.cuh"

typedef struct {
    unsigned long long seed;
    long long photons;
    long long chunk_size;
    long long first_chunk;
    long long last_chunk;
} PhotonWork;

typedef struct {
    unsigned long long next_chunk;
    unsigned long long simulated;
    unsigned long long blocked_by_spots;
    unsigned long long missed_observers;
    unsigned long long recorded;
} GpuCounters;

typedef struct {
    PhotonCounts tallies;
    unsigned long long chunks;
} ThreadWork;

typedef struct {
    int threads_per_block;
    int blocks_per_sm;
    int blocks;
    bool cooperative;
} Launch;

struct Processor {
    const Options *options;
    bool record;
    GpuImage image;
    PixelCount *host_counts;
    GpuCounters *counters;
    SmWork *sm_work;
    Launch launch;
};

__constant__ Scene gpu_scene;


// ============================================================ the kernel

__device__ long long smaller(long long a, long long b) {
    return a < b ? a : b;
}

__device__ long long take_next_chunk(const PhotonWork work, GpuCounters *counters) {
    return work.first_chunk + (long long) atomicAdd(&counters->next_chunk, 1ULL);
}

__device__ void trace_chunk(long long chunk, const PhotonWork work, PixelCount *image, PhotonCounts *tallies) {
    GpuRng rng;
    curand_init(work.seed, (unsigned long long) chunk, 0, &rng);
    long long first_photon = chunk * work.chunk_size;
    long long end_photon = smaller(first_photon + work.chunk_size, work.photons);
    for (long long photon = first_photon; photon < end_photon; photon++) {
        long long pixel = trace_photon(&gpu_scene, &rng);
        if (pixel == PHOTON_BLOCKED_BY_SPOT) {
            tallies->blocked_by_spots++;
        } else if (pixel == PHOTON_MISSES_OBSERVERS) {
            tallies->missed_observers++;
        } else {
            tallies->recorded++;
            if (image != NULL) {
                count_photon_atomic(image, (unsigned long long) pixel);
            }
        }
    }
    tallies->simulated += end_photon - first_photon;
}

__device__ ThreadWork take_chunks_until_none_left(const PhotonWork work, PixelCount *image, GpuCounters *counters) {
    ThreadWork done = {{0, 0, 0, 0}, 0};
    long long chunk = take_next_chunk(work, counters);
    while (chunk < work.last_chunk) {
        trace_chunk(chunk, work, image, &done.tallies);
        done.chunks++;
        chunk = take_next_chunk(work, counters);
    }
    return done;
}

__device__ void add_tallies(GpuCounters *counters, PhotonCounts tallies) {
    atomicAdd(&counters->simulated, (unsigned long long) tallies.simulated);
    atomicAdd(&counters->blocked_by_spots, (unsigned long long) tallies.blocked_by_spots);
    atomicAdd(&counters->missed_observers, (unsigned long long) tallies.missed_observers);
    atomicAdd(&counters->recorded, (unsigned long long) tallies.recorded);
}

__global__ void trace_photons(const PhotonWork work, PixelCount *image, GpuCounters *counters, SmWork *sm_work) {
    long long started_at = clock64();
    unsigned int sm = sm_id();
    if (!is_allowed(sm, sm_work)) {
        return;
    }
    ThreadWork done = take_chunks_until_none_left(work, image, counters);
    add_tallies(counters, done.tallies);
    add_sm_work(sm_work, sm, done.chunks, (unsigned long long) done.tallies.simulated,
                (unsigned long long) done.tallies.recorded, (unsigned long long) (clock64() - started_at));
}


// ============================================================ the launch

static Launch choose_launch(void) {
    Launch launch;
    int smallest_full_grid;
    check(cudaOccupancyMaxPotentialBlockSize(&smallest_full_grid, &launch.threads_per_block, trace_photons, 0, 0),
          "choosing the block size");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&launch.blocks_per_sm, trace_photons,
                                                        launch.threads_per_block, 0),
          "counting blocks per SM");
    launch.blocks = launch.blocks_per_sm * gpu_attribute(cudaDevAttrMultiProcessorCount);
    launch.cooperative = gpu_attribute(cudaDevAttrCooperativeLaunch) == 1;
    return launch;
}

static void run_kernel(const void *kernel, Launch launch, void **arguments) {
    if (launch.cooperative) {
        check(cudaLaunchCooperativeKernel(kernel, launch.blocks, launch.threads_per_block, arguments, 0, 0),
              "launching the kernel");
    } else {
        check(cudaLaunchKernel(kernel, launch.blocks, launch.threads_per_block, arguments, 0, 0),
              "launching the kernel");
    }
    check(cudaDeviceSynchronize(), "waiting for the kernel");
}

static PhotonWork photon_work(const Options *options) {
    PhotonWork work = {options->star.seed, options->star.photons, options->chunk_size,
                       options->first_chunk, options->last_chunk};
    return work;
}

static PhotonCounts read_counters(const GpuCounters *gpu_counters) {
    GpuCounters counters;
    check(cudaMemcpy(&counters, gpu_counters, sizeof(counters), cudaMemcpyDeviceToHost), "reading the counters");
    PhotonCounts counts = {(long long) counters.simulated, (long long) counters.blocked_by_spots,
                           (long long) counters.missed_observers, (long long) counters.recorded};
    return counts;
}


// ============================================================ the processor

static KernelResources describe_kernel(Launch launch, int max_threads_per_sm) {
    cudaFuncAttributes attributes;
    check(cudaFuncGetAttributes(&attributes, trace_photons), "reading the kernel's attributes");
    KernelResources kernel;
    kernel.registers_per_thread = attributes.numRegs;
    kernel.local_bytes_per_thread = attributes.localSizeBytes;
    kernel.constant_bytes = attributes.constSizeBytes;
    kernel.threads_per_block = launch.threads_per_block;
    kernel.blocks_per_unit = launch.blocks_per_sm;
    kernel.blocks = launch.blocks;
    kernel.occupancy = (double) launch.blocks_per_sm * launch.threads_per_block / max_threads_per_sm;
    return kernel;
}

void describe_processor(const Options *options, ProcessorInfo *info) {
    describe_gpu(info);
    info->compute_units = resolve_sm_count(options->sms);
    info->kernel = describe_kernel(choose_launch(), info->hardware.max_threads_per_unit);
}

Processor *open_processor(const Options *options) {
    Processor *processor = (Processor *) calloc(1, sizeof(Processor));
    processor->options = options;
    processor->record = options->record;
    if (processor->record) {
        size_t pixels = pixels_per_observer(&options->star);
        allocate_gpu_image(&processor->image, pixels, options->observers_per_pass);
        zero_gpu_image(&processor->image, options->observers_per_pass);
        check(cudaMallocHost(&processor->host_counts, pixels * sizeof(PixelCount)), "allocating the host buffer");
    }
    check(cudaMalloc(&processor->counters, sizeof(GpuCounters)), "allocating the counters");
    processor->launch = choose_launch();
    processor->sm_work = create_sm_work(resolve_sm_count(options->sms), processor->launch.blocks,
                                        processor->launch.threads_per_block);
    return processor;
}

void load_pass(Processor *processor, const Scene *scene) {
    check(cudaMemcpyToSymbol(gpu_scene, scene, sizeof(Scene)), "copying the scene to constant memory");
}

void zero_pass_image(Processor *processor, unsigned int observers) {
    if (processor->record) {
        zero_gpu_image(&processor->image, observers);
    }
}

PhotonCounts trace_pass(Processor *processor, const Run *run, Progress *progress) {
    PhotonWork work = photon_work(run->options);
    PixelCount *image = processor->record ? processor->image.counts : NULL;
    void *arguments[] = {&work, &image, &processor->counters, &processor->sm_work};
    check(cudaMemset(processor->counters, 0, sizeof(GpuCounters)), "resetting the counters");
    reset_sm_work(processor->sm_work);
    run_kernel((const void *) trace_photons, processor->launch, arguments);
    print_sm_work(processor->sm_work, progress->pass);
    return read_counters(processor->counters);
}

const PixelCount *observer_on_host(Processor *processor, unsigned int slot) {
    copy_observer_to_host(&processor->image, slot, processor->host_counts);
    return processor->host_counts;
}

size_t processor_image_bytes(const Processor *processor) {
    return processor->record ? gpu_image_bytes(&processor->image) : 0;
}

void close_processor(Processor *processor) {
    if (processor->record) {
        free_gpu_image(&processor->image);
        cudaFreeHost(processor->host_counts);
    }
    cudaFree(processor->counters);
    free_sm_work(processor->sm_work);
    free(processor);
}
