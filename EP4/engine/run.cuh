#ifndef RUN_CUH
#define RUN_CUH

#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "options.cuh"
#include "observers.cuh"

#define DEVICE_NAME "CPU"
#define HOST_NAME_LENGTH 256

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

static inline void read_host_name(char *host) {
    if (gethostname(host, HOST_NAME_LENGTH) != 0) {
        snprintf(host, HOST_NAME_LENGTH, "unknown");
    }
    host[HOST_NAME_LENGTH - 1] = '\0';
}

#endif
