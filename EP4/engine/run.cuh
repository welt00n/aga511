#ifndef RUN_CUH
#define RUN_CUH

#include <stddef.h>
#include <time.h>

#include "options.cuh"
#include "observers.cuh"

#define DEVICE_NAME "CPU"

typedef struct {
    long long simulated;
    long long blocked_by_spots;
    long long missed_observers;
    long long recorded;
} PhotonCounts;

typedef struct {
    double zero;
    double photons;
    double save;
} PassTimes;

typedef struct {
    PhotonCounts counts;
    PassTimes times;
} PassResult;

typedef struct {
    unsigned long long recorded;
    size_t bytes;
    double save_time;
} ObserverResult;

typedef struct {
    double setup;
    double zero;
    double photons;
    double save;
    double total;
} RunTimes;

typedef struct {
    const Options *options;
    const ObserverList *observers;
    unsigned int passes;
    double started_at;
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

#endif
