#include <stdio.h>
#include <stdlib.h>

#include "processor.cuh"
#include "physics/photon.cuh"
#include "gpu/gpu_device.cuh"
#include "gpu/gpu_image.cuh"
#include "random/gpu_rng.cuh"
#include "gpu/gpu_sms.cuh"
#include "gpu/gpu_sections.cuh"

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
    KernelLaps laps;
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
    unsigned long long *section_cycles;
    Launch launch;
};

#ifdef STAR_SCENE_IN_GLOBAL
__device__ Scene gpu_scene;
static const char *SCENE_MEMORY = "global";
#else
__constant__ Scene gpu_scene;
static const char *SCENE_MEMORY = "constant";
#endif


// ============================================================ the kernel

__device__ long long smaller(long long a, long long b) {
    return a < b ? a : b;
}

__device__ long long take_next_chunk(const PhotonWork work, GpuCounters *counters) {
    return work.first_chunk + (long long) atomicAdd(&counters->next_chunk, 1ULL);
}

__device__ void tally_photon(long long pixel, PixelCount *image, PhotonCounts *tallies) {
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

__device__ void trace_chunk(long long chunk, const PhotonWork work, PixelCount *image, ThreadWork *done) {
    GpuRng rng;
    curand_init(work.seed, (unsigned long long) chunk, 0, &rng);
    done->laps.lap(SECTION_CHUNK);
    long long first_photon = chunk * work.chunk_size;
    long long end_photon = smaller(first_photon + work.chunk_size, work.photons);
    for (long long photon = first_photon; photon < end_photon; photon++) {
        long long pixel = trace_photon(&gpu_scene, &rng, &done->laps);
        tally_photon(pixel, image, &done->tallies);
        done->laps.lap(SECTION_RECORD);
    }
    done->tallies.simulated += end_photon - first_photon;
}

__device__ ThreadWork take_chunks_until_none_left(const PhotonWork work, PixelCount *image, GpuCounters *counters) {
    ThreadWork done = {};
    done.laps.start();
    long long chunk = take_next_chunk(work, counters);
    while (chunk < work.last_chunk) {
        trace_chunk(chunk, work, image, &done);
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

__global__ void trace_photons(const PhotonWork work, PixelCount *image, GpuCounters *counters, SmWork *sm_work,
                              unsigned long long *section_cycles) {
    long long started_at = clock64();
    unsigned int sm = sm_id();
    if (!is_allowed(sm, sm_work)) {
        return;
    }
    ThreadWork done = take_chunks_until_none_left(work, image, counters);
    add_tallies(counters, done.tallies);
    add_sm_work(sm_work, sm, done.chunks, (unsigned long long) done.tallies.simulated,
                (unsigned long long) done.tallies.recorded, (unsigned long long) (clock64() - started_at));
    done.laps.add_to(section_cycles);
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

static double run_kernel(const void *kernel, Launch launch, void **arguments) {
    cudaEvent_t started, finished;
    check(cudaEventCreate(&started), "creating an event");
    check(cudaEventCreate(&finished), "creating an event");
    check(cudaEventRecord(started), "recording the kernel's start");
    if (launch.cooperative) {
        check(cudaLaunchCooperativeKernel(kernel, launch.blocks, launch.threads_per_block, arguments, 0, 0),
              "launching the kernel");
    } else {
        check(cudaLaunchKernel(kernel, launch.blocks, launch.threads_per_block, arguments, 0, 0),
              "launching the kernel");
    }
    check(cudaEventRecord(finished), "recording the kernel's end");
    check(cudaEventSynchronize(finished), "waiting for the kernel");
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, started, finished), "timing the kernel");
    cudaEventDestroy(started);
    cudaEventDestroy(finished);
    return milliseconds / 1e3;
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
    if (SECTION_TIMERS) {
        check(cudaMalloc(&processor->section_cycles, SECTIONS * sizeof(unsigned long long)), "allocating the timers");
    }
    return processor;
}

void load_pass(Processor *processor, const Scene *scene) {
    check(cudaMemcpyToSymbol(gpu_scene, scene, sizeof(Scene)), "copying the scene to the GPU");
}

void zero_pass_image(Processor *processor, unsigned int observers) {
    if (processor->record) {
        zero_gpu_image(&processor->image, observers);
    }
}

PhotonCounts trace_pass(Processor *processor, const Run *run, Progress *progress) {
    PhotonWork work = photon_work(run->options);
    PixelCount *image = processor->record ? processor->image.counts : NULL;
    void *arguments[] = {&work, &image, &processor->counters, &processor->sm_work, &processor->section_cycles};
    check(cudaMemset(processor->counters, 0, sizeof(GpuCounters)), "resetting the counters");
    reset_sm_work(processor->sm_work);
    if (SECTION_TIMERS) {
        check(cudaMemset(processor->section_cycles, 0, SECTIONS * sizeof(unsigned long long)), "resetting the timers");
    }
    double kernel_time = run_kernel((const void *) trace_photons, processor->launch, arguments);
    printf("KERNEL PASS=%u KERNEL_TIME=%.6f SCENE_MEMORY=%s SECTION_TIMERS=%d\n", progress->pass, kernel_time,
           SCENE_MEMORY, SECTION_TIMERS ? 1 : 0);
    print_sm_work(processor->sm_work, progress->pass);
    if (SECTION_TIMERS) {
        print_sections(processor->section_cycles, progress->pass);
    }
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
    cudaFree(processor->section_cycles);
    free_sm_work(processor->sm_work);
    free(processor);
}
