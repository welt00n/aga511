#include <stdio.h>

#include "processor.cuh"
#include "gpu_device.cuh"

struct Processor {
    const Options *options;
};

void describe_processor(const Options *options, ProcessorInfo *info) {
    describe_gpu(info);
}

int compare_rng_with_cpu(unsigned long long seed) {
    return 0;
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
