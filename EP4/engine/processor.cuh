#ifndef PROCESSOR_CUH
#define PROCESSOR_CUH

#include <stddef.h>

#include "options.cuh"
#include "scene.cuh"
#include "image.cuh"
#include "run.cuh"

typedef struct Processor Processor;

void describe_processor(const Options *options, ProcessorInfo *info);
int compare_rng_with_cpu(unsigned long long seed);

Processor *open_processor(const Options *options);
void load_pass(Processor *processor, const Scene *scene);
void zero_pass_image(Processor *processor, unsigned int observers);
PhotonCounts trace_pass(Processor *processor, const Run *run, Progress *progress);
const PixelCount *observer_on_host(Processor *processor, unsigned int slot);
size_t processor_image_bytes(const Processor *processor);
void close_processor(Processor *processor);

#endif
