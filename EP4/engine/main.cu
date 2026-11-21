#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>

#include "real.cuh"
#include "rng.cuh"
#include "scene.cuh"
#include "photon.cuh"
#include "image.cuh"
#include "options.cuh"
#include "observers.cuh"
#include "run.cuh"
#include "report.cuh"
#include "files.cuh"

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

static Scene scene;
static ObserverList observers;


// ============================================================ helpers

static long long smaller(long long a, long long b) {
    return a < b ? a : b;
}

static int prepare_output_dir(const char *folder) {
    if (mkdir(folder, 0755) != 0 && errno != EEXIST) {
        perror("Could not create the output folder");
        return -1;
    }
    return 0;
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

static PhotonCounts simulate_chunks(const Run *run, Image *image, Progress *progress) {
    const Options *options = run->options;
    PhotonCounts counts = {0, 0, 0, 0};
    long long next_chunk = options->first_chunk;
    while (next_chunk < options->last_chunk) {
        simulate_chunk(&scene, next_chunk, options, image, &counts);
        next_chunk++;
        progress->chunks_done++;
        report_progress_if_due(run, progress);
    }
    return counts;
}


// ============================================================ passes

static int save_pass_observers(const Run *run, const Pass *pass, const Image *image, const PassResult *pass_result) {
    for (unsigned int slot = 0; slot < pass->observer_count; slot++) {
        const Observer *observer = &run->observers->observers[pass->first_observer + slot];
        char path[PATH_LENGTH];
        observer_path(run->options->output_dir, observer, ".bin", path);
        double started_at = now_in_seconds();
        if (save_observer(image, slot, path) != 0) {
            return -1;
        }
        ObserverResult saved = {photons_in_observer(image, slot), observer_bytes(image), 0};
        saved.save_time = now_in_seconds() - started_at;
        if (write_observer_record(run, pass, observer, pass_result, &saved) != 0) {
            return -1;
        }
    }
    return 0;
}

static int run_pass(const Run *run, const Pass *pass, Image *image, PassResult *result) {
    bool record = run->options->record;
    assign_pass_observers(&scene, run->observers, pass);

    double phase_started_at = now_in_seconds();
    if (record) {
        zero_image(image, pass->observer_count);
    }
    result->times.zero = now_in_seconds() - phase_started_at;

    phase_started_at = now_in_seconds();
    Progress progress = {pass->number, 0, phase_started_at, phase_started_at};
    result->counts = simulate_chunks(run, record ? image : NULL, &progress);
    result->times.photons = now_in_seconds() - phase_started_at;

    phase_started_at = now_in_seconds();
    if (record && save_pass_observers(run, pass, image, result) != 0) {
        return -1;
    }
    result->times.save = now_in_seconds() - phase_started_at;
    print_pass(run, pass, result);
    return 0;
}

static void add_pass(PhotonCounts *counts, RunTimes *times, const PassResult *pass) {
    counts->simulated = pass->counts.simulated;
    counts->blocked_by_spots = pass->counts.blocked_by_spots;
    counts->recorded += pass->counts.recorded;
    counts->missed_observers = counts->simulated - counts->blocked_by_spots - counts->recorded;
    times->zero += pass->times.zero;
    times->photons += pass->times.photons;
    times->save += pass->times.save;
}


// ============================================================ main

static int prepare_run(const Options *options, Image *image) {
    if (build_scene(&options->star, &scene) != 0 || locate_observers(&options->star, &observers) != 0) {
        return -1;
    }
    if (!options->record) {
        return 0;
    }
    if (prepare_output_dir(options->output_dir) != 0) {
        return -1;
    }
    return allocate_image(image, pixels_per_observer(&options->star), options->observers_per_pass);
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    double started_at = now_in_seconds();
    Options options = load_options(argc, argv);
    if (options.test_rng) {
        return test_rng();
    }
    RunTimes times = {0, 0, 0, 0, 0};
    Image image = {NULL, 0, 0};
    if (prepare_run(&options, &image) != 0) {
        return 1;
    }
    times.setup = now_in_seconds() - started_at;

    Run run = {&options, &observers, pass_count(&options), started_at};
    PhotonCounts counts = {0, 0, 0, 0};
    for (unsigned int index = 0; index < run.passes; index++) {
        Pass pass = make_pass(index, options.observers_per_pass, observers.count);
        PassResult result;
        if (run_pass(&run, &pass, &image, &result) != 0) {
            return 1;
        }
        add_pass(&counts, &times, &result);
    }

    size_t allocated_bytes = options.record ? image_bytes(&image) : 0;
    free_image(&image);
    times.total = now_in_seconds() - started_at;
    print_summary(&run, &counts, &times, allocated_bytes);
    return 0;
}
