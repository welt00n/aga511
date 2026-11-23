#ifndef RUN_CUH
#define RUN_CUH

#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "run/options.cuh"
#include "physics/observers.cuh"

#define HOST_NAME_LENGTH 256
#define PROCESSOR_NAME_LENGTH 256
#define COMPUTE_CAPABILITY_LENGTH 16

typedef struct {
    long long simulated;
    long long blocked_by_spots;
    long long missed_observers;
    long long recorded;
} PhotonCounts;

typedef struct {
    double zero;
    double photons;
    double copy;
    double save;
} PassTimes;

typedef struct {
    PhotonCounts counts;
    PassTimes times;
} PassResult;

typedef struct {
    unsigned long long recorded;
    size_t bytes;
    double copy_time;
    double save_time;
} ObserverResult;

typedef struct {
    double setup;
    double zero;
    double photons;
    double copy;
    double save;
    double total;
} RunTimes;

typedef struct {
    size_t memory_total;
    size_t memory_free;
    size_t memory_for_image;
    size_t l2_bytes;
    double peak_bandwidth;
    int fp32_per_fp64;
    int max_threads_per_unit;
    int runtime_version;
    int driver_version;
} ProcessorHardware;

typedef struct {
    int registers_per_thread;
    size_t local_bytes_per_thread;
    size_t constant_bytes;
    int threads_per_block;
    int blocks_per_unit;
    int blocks;
    double occupancy;
} KernelResources;

typedef struct {
    const char *kind;
    char name[PROCESSOR_NAME_LENGTH];
    char compute_capability[COMPUTE_CAPABILITY_LENGTH];
    unsigned int compute_units;
    unsigned int compute_units_total;
    const char *unit;
    ProcessorHardware hardware;
    KernelResources kernel;
} ProcessorInfo;

typedef struct {
    const Options *options;
    const ObserverList *observers;
    unsigned int passes;
    double started_at;
    const ProcessorInfo *processor;
    char host[HOST_NAME_LENGTH];
} Run;

typedef struct {
    unsigned int pass;
    long long chunks_done;
    double pass_started_at;
    double last_printed_at;
} Progress;

static inline double now_in_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static inline void replace_spaces_with_underscores(char *text) {
    for (char *c = text; *c; c++) {
        if (*c == ' ') {
            *c = '_';
        }
    }
}

static inline void read_host_name(char *host) {
    if (gethostname(host, HOST_NAME_LENGTH) != 0) {
        snprintf(host, HOST_NAME_LENGTH, "unknown");
    }
    host[HOST_NAME_LENGTH - 1] = '\0';
}

#endif
