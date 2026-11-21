#ifndef OPTIONS_CUH
#define OPTIONS_CUH

#include <stddef.h>

#include "scene.cuh"

typedef struct {
    StarConfig star;
    long long chunk_size;
    long long first_chunk;
    long long last_chunk;
    unsigned int observers_per_pass;
    double memory_budget;
    bool record;
    const char *output_dir;
    const char *fingerprint;
    const char *code_hash;
    bool test_rng;
} Options;

Options load_options(int argc, char *argv[]);
long long chunk_count(const Options *options);
unsigned int pass_count(const Options *options);
size_t pixels_per_observer(const StarConfig *star);

#endif
