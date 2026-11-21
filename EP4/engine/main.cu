#include <stdio.h>
#include <time.h>

#include "real.cuh"
#include "rng.cuh"
#include "scene.cuh"
#include "photon.cuh"
#include "image.cuh"
#include "options.cuh"

typedef struct {
    long long simulated;
    long long blocked_by_spots;
    long long missed_observers;
    long long recorded;
} PhotonCounts;

typedef struct {
    double setup;
    double photons;
    double save;
    double total;
} PhaseTimes;

typedef struct {
    PhiloxCounter counter;
    PhiloxKey key;
    PhiloxCounter expected;
} PhiloxKnownAnswer;

static const PhiloxKnownAnswer RANDOM123_KNOWN_ANSWERS[] = {
    {{0x00000000, 0x00000000, 0x00000000, 0x00000000}, {0x00000000, 0x00000000},
     {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
    {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff},
     {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
    {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0},
     {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
};


// ============================================================ helpers

static double now_in_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static long long smaller(long long a, long long b) {
    return a < b ? a : b;
}


// ============================================================ random numbers

static bool same_counter(PhiloxCounter a, PhiloxCounter b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

static void print_first_numbers_of_chunk_0(unsigned long long seed) {
    PhiloxState rng;
    philox_init(seed, 0, 0, &rng);
    printf("SEED_%llu_CHUNK_0=", seed);
    for (int i = 0; i < 8; i++) {
        printf("%08x%s", philox_next(&rng), i < 7 ? "," : "\n");
    }
}

static int test_rng(void) {
    int failures = 0;
    int tests = sizeof(RANDOM123_KNOWN_ANSWERS) / sizeof(RANDOM123_KNOWN_ANSWERS[0]);
    for (int i = 0; i < tests; i++) {
        PhiloxKnownAnswer test = RANDOM123_KNOWN_ANSWERS[i];
        PhiloxCounter got = philox4x32_10(test.counter, test.key);
        bool passed = same_counter(got, test.expected);
        printf("PHILOX_TEST_%d=%s GOT=%08x,%08x,%08x,%08x\n", i, passed ? "PASS" : "FAIL", got.x, got.y, got.z, got.w);
        failures += passed ? 0 : 1;
    }
    print_first_numbers_of_chunk_0(42);
    printf("PHILOX_TESTS=%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}


// ============================================================ photons

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

static PhotonCounts simulate_chunks(const Scene *scene, const Options *options, Image *image) {
    PhotonCounts counts = {0, 0, 0, 0};
    long long next_chunk = options->first_chunk;
    while (next_chunk < options->last_chunk) {
        simulate_chunk(scene, next_chunk, options, image, &counts);
        next_chunk++;
    }
    return counts;
}


// ============================================================ result

static void print_result(const Options *options, const PhotonCounts *counts, const PhaseTimes *times,
                         size_t image_size) {
    const StarConfig *star = &options->star;
    printf("DEVICE=CPU PRECISION=%d N=%lld SEED=%llu A=%g SPOTS=%u UMBRA_RADIUS=%g PENUMBRA_RADIUS=%g ",
           PRECISION_BITS, star->photons, star->seed, star->limb_darkening, star->spots,
           star->umbra_radius, star->penumbra_radius);
    printf("OBSERVERS=%u NX=%u NY=%u NPHI=%u NMU=%u ", star->observers, star->nx, star->ny, star->nphi, star->nmu);
    printf("CHUNK_SIZE=%lld CHUNKS=%lld FIRST_CHUNK=%lld LAST_CHUNK=%lld ",
           options->chunk_size, chunk_count(options), options->first_chunk, options->last_chunk);
    printf("PHOTONS=%lld BLOCKED=%lld MISSED=%lld RECORDED=%lld RECORD=%d IMAGE_BYTES=%zu ",
           counts->simulated, counts->blocked_by_spots, counts->missed_observers, counts->recorded,
           options->record ? 1 : 0, image_size);
    printf("SETUP_TIME=%.6f PHOTONS_TIME=%.6f SAVE_TIME=%.6f TIME=%.6f PHOTONS_PER_SECOND=%.6e IMAGE=%s\n",
           times->setup, times->photons, times->save, times->total,
           times->photons > 0 ? counts->simulated / times->photons : 0.0,
           options->record ? options->output : "none");
}


// ============================================================ main

int main(int argc, char *argv[]) {
    double started_at = now_in_seconds();
    Options options = load_options(argc, argv);
    if (options.test_rng) {
        return test_rng();
    }
    PhaseTimes times;

    double phase_started_at = now_in_seconds();
    static Scene scene;
    if (build_scene(&options.star, &scene) != 0) {
        return 1;
    }
    Image image = {NULL, 0};
    if (options.record && allocate_image(&image, options.star.nx, options.star.ny,
                                         options.star.nphi, options.star.observers) != 0) {
        return 1;
    }
    times.setup = now_in_seconds() - phase_started_at;

    phase_started_at = now_in_seconds();
    PhotonCounts counts = simulate_chunks(&scene, &options, options.record ? &image : NULL);
    times.photons = now_in_seconds() - phase_started_at;

    phase_started_at = now_in_seconds();
    if (options.record && save_image(&image, options.output) != 0) {
        return 1;
    }
    times.save = now_in_seconds() - phase_started_at;
    size_t image_size = options.record ? image_bytes(&image) : 0;
    free_image(&image);

    times.total = now_in_seconds() - started_at;
    print_result(&options, &counts, &times, image_size);
    return 0;
}
