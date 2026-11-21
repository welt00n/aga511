#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "processor.cuh"
#include "photon.cuh"
#include "rng.cuh"
#include "report.cuh"

#define CPU_INFO_LINE_LENGTH 512

struct Processor {
    const Options *options;
    const Scene *scene;
    bool record;
    Image image;
};

static long long smaller(long long a, long long b) {
    return a < b ? a : b;
}

static void read_cpu_name(char *name) {
    snprintf(name, PROCESSOR_NAME_LENGTH, "unknown");
    FILE *file = fopen("/proc/cpuinfo", "r");
    if (file == NULL) {
        return;
    }
    char line[CPU_INFO_LINE_LENGTH];
    while (fgets(line, sizeof(line), file) != NULL) {
        char *colon = strchr(line, ':');
        if (strncmp(line, "model name", 10) == 0 && colon != NULL) {
            snprintf(name, PROCESSOR_NAME_LENGTH, "%s", colon + 2);
            name[strcspn(name, "\n")] = '\0';
            break;
        }
    }
    fclose(file);
    replace_spaces_with_underscores(name);
}

void describe_processor(const Options *options, ProcessorInfo *info) {
    info->kind = "CPU";
    read_cpu_name(info->name);
    snprintf(info->compute_capability, COMPUTE_CAPABILITY_LENGTH, "none");
    info->compute_units = 1;
    info->unit = "core";
}

int compare_rng_with_cpu(unsigned long long seed) {
    return 0;
}

Processor *open_processor(const Options *options) {
    Processor *processor = (Processor *) calloc(1, sizeof(Processor));
    processor->options = options;
    processor->record = options->record;
    if (processor->record &&
        allocate_image(&processor->image, pixels_per_observer(&options->star), options->observers_per_pass) != 0) {
        free(processor);
        return NULL;
    }
    return processor;
}

void load_pass(Processor *processor, const Scene *scene) {
    processor->scene = scene;
}

void zero_pass_image(Processor *processor, unsigned int observers) {
    if (processor->record) {
        zero_image(&processor->image, observers);
    }
}

static void simulate_chunk(const Scene *scene, long long chunk, const Options *options,
                           Image *image, PhotonCounts *counts) {
    PhiloxState rng;
    philox_init(options->star.seed, (unsigned long long) chunk, 0, &rng);
    long long first_photon = chunk * options->chunk_size;
    long long end_photon = smaller(first_photon + options->chunk_size, options->star.photons);
    for (long long photon = first_photon; photon < end_photon; photon++) {
        long long pixel = trace_photon(scene, &rng);
        if (pixel == PHOTON_BLOCKED_BY_SPOT) {
            counts->blocked_by_spots++;
        } else if (pixel == PHOTON_MISSES_OBSERVERS) {
            counts->missed_observers++;
        } else {
            counts->recorded++;
            if (image != NULL) {
                count_photon(image, (unsigned long long) pixel);
            }
        }
    }
    counts->simulated += end_photon - first_photon;
}

PhotonCounts trace_pass(Processor *processor, const Run *run, Progress *progress) {
    const Options *options = run->options;
    Image *image = processor->record ? &processor->image : NULL;
    PhotonCounts counts = {0, 0, 0, 0};
    long long next_chunk = options->first_chunk;
    while (next_chunk < options->last_chunk) {
        simulate_chunk(processor->scene, next_chunk, options, image, &counts);
        next_chunk++;
        progress->chunks_done++;
        report_progress_if_due(run, progress);
    }
    return counts;
}

const PixelCount *observer_on_host(Processor *processor, unsigned int slot) {
    return observer_counts(&processor->image, slot);
}

size_t processor_image_bytes(const Processor *processor) {
    return processor->record ? image_bytes(&processor->image) : 0;
}

void close_processor(Processor *processor) {
    if (processor->record) {
        free_image(&processor->image);
    }
    free(processor);
}
