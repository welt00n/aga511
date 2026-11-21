#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>

#include "rng.cuh"
#include "scene.cuh"
#include "image.cuh"
#include "options.cuh"
#include "observers.cuh"
#include "processor.cuh"
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
static ProcessorInfo processor_info;


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
    failures += compare_rng_with_cpu(42);
    printf("PHILOX_TESTS=%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}


// ============================================================ passes

static int prepare_output_dir(const char *folder) {
    if (mkdir(folder, 0755) != 0 && errno != EEXIST) {
        perror("Could not create the output folder");
        return -1;
    }
    return 0;
}

static int save_observer_files(const Run *run, const Pass *pass, const Observer *observer, const PixelCount *counts,
                               const PassResult *pass_result, ObserverResult *saved) {
    size_t pixels = pixels_per_observer(&run->options->star);
    char path[PATH_LENGTH];
    observer_path(run->options->output_dir, observer, ".bin", path);
    double started_at = now_in_seconds();
    if (save_counts(counts, pixels, path) != 0) {
        return -1;
    }
    saved->recorded = photons_in(counts, pixels);
    saved->bytes = pixels * sizeof(PixelCount);
    saved->save_time = now_in_seconds() - started_at;
    return write_observer_record(run, pass, observer, pass_result, saved);
}

static int save_pass_observers(const Run *run, const Pass *pass, Processor *processor, PassResult *result) {
    for (unsigned int slot = 0; slot < pass->observer_count; slot++) {
        const Observer *observer = &run->observers->observers[pass->first_observer + slot];
        double started_at = now_in_seconds();
        const PixelCount *counts = observer_on_host(processor, slot);
        ObserverResult saved = {0, 0, now_in_seconds() - started_at, 0};
        result->times.copy += saved.copy_time;
        if (save_observer_files(run, pass, observer, counts, result, &saved) != 0) {
            return -1;
        }
    }
    return 0;
}

static int run_pass(const Run *run, const Pass *pass, Processor *processor, PassResult *result) {
    assign_pass_observers(&scene, run->observers, pass);
    load_pass(processor, &scene);

    double phase_started_at = now_in_seconds();
    zero_pass_image(processor, pass->observer_count);
    result->times.zero = now_in_seconds() - phase_started_at;

    phase_started_at = now_in_seconds();
    Progress progress = {pass->number, 0, phase_started_at, phase_started_at};
    result->counts = trace_pass(processor, run, &progress);
    result->times.photons = now_in_seconds() - phase_started_at;

    result->times.copy = 0;
    phase_started_at = now_in_seconds();
    if (run->options->record && save_pass_observers(run, pass, processor, result) != 0) {
        return -1;
    }
    result->times.save = now_in_seconds() - phase_started_at - result->times.copy;
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
    times->copy += pass->times.copy;
    times->save += pass->times.save;
}


// ============================================================ main

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    double started_at = now_in_seconds();
    Options options = load_options(argc, argv);
    if (options.test_rng) {
        return test_rng();
    }
    if (build_scene(&options.star, &scene) != 0 || locate_observers(&options.star, &observers) != 0) {
        return 1;
    }
    describe_processor(&options, &processor_info);
    Run run = {&options, &observers, pass_count(&options), started_at, &processor_info, ""};
    read_host_name(run.host);
    if (options.describe) {
        print_config(&run);
        return 0;
    }

    if (options.record && prepare_output_dir(options.output_dir) != 0) {
        return 1;
    }
    Processor *processor = open_processor(&options);
    if (processor == NULL) {
        return 1;
    }
    RunTimes times = {now_in_seconds() - started_at, 0, 0, 0, 0, 0};

    PhotonCounts counts = {0, 0, 0, 0};
    for (unsigned int index = 0; index < run.passes; index++) {
        Pass pass = make_pass(index, options.observers_per_pass, observers.count);
        PassResult result;
        if (run_pass(&run, &pass, processor, &result) != 0) {
            return 1;
        }
        add_pass(&counts, &times, &result);
    }

    size_t allocated_bytes = processor_image_bytes(processor);
    close_processor(processor);
    times.total = now_in_seconds() - started_at;
    print_summary(&run, &counts, &times, allocated_bytes);
    return 0;
}
