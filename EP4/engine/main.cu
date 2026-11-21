#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>

#include "scene.cuh"
#include "image.cuh"
#include "options.cuh"
#include "observers.cuh"
#include "processor.cuh"
#include "rng_test.cuh"
#include "run.cuh"
#include "report.cuh"
#include "files.cuh"

static Scene scene;
static ObserverList observers;
static ProcessorInfo processor_info;


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
