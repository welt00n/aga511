#ifndef GPU_SECTIONS_CUH
#define GPU_SECTIONS_CUH

#include <stdio.h>

#include "photon.cuh"
#include "gpu_device.cuh"

enum { SECTION_CHUNK = PHOTON_SECTIONS, SECTION_RECORD, SECTIONS };

static const char *SECTION_NAMES[SECTIONS] = {"surface", "direction", "spots", "bin", "pixel", "chunk", "record"};

struct SilentLaps {
    __device__ void start() {}
    __device__ void lap(int section) {}
    __device__ void add_to(unsigned long long *totals) {}
};

struct ClockLaps {
    long long last;
    unsigned long long cycles[SECTIONS];

    __device__ void start() {
        last = clock64();
    }

    __device__ void lap(int section) {
        long long now = clock64();
        cycles[section] += now - last;
        last = now;
    }

    __device__ void add_to(unsigned long long *totals) {
        for (int section = 0; section < SECTIONS; section++) {
            atomicAdd(&totals[section], cycles[section]);
        }
    }
};

#ifdef STAR_SECTION_TIMERS
typedef ClockLaps KernelLaps;
static const bool SECTION_TIMERS = true;
#else
typedef SilentLaps KernelLaps;
static const bool SECTION_TIMERS = false;
#endif

static inline void print_sections(const unsigned long long *gpu_totals, unsigned int pass) {
    unsigned long long totals[SECTIONS];
    check(cudaMemcpy(totals, gpu_totals, sizeof(totals), cudaMemcpyDeviceToHost), "reading the section timers");
    unsigned long long all = 0;
    for (int section = 0; section < SECTIONS; section++) {
        all += totals[section];
    }
    for (int section = 0; section < SECTIONS; section++) {
        printf("SECTION PASS=%u NAME=%s CYCLES=%llu SHARE=%.4f\n", pass, SECTION_NAMES[section], totals[section],
               all > 0 ? (double) totals[section] / all : 0.0);
    }
}

#endif
