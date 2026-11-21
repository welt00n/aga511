#ifndef OPTIONS_CUH
#define OPTIONS_CUH

#include "scene.cuh"

typedef struct {
    StarConfig star;
    long long chunk_size;
    long long first_chunk;
    long long last_chunk;
    bool record;
    const char *output;
    bool test_rng;
} Options;

Options load_options(int argc, char *argv[]);
long long chunk_count(const Options *options);

#endif
