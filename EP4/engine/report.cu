#include <stdio.h>
#include <time.h>

#include "report.cuh"
#include "real.cuh"
#include "image.cuh"
#include "files.cuh"

#define PROGRESS_INTERVAL_SECONDS 10.0
#define TIMESTAMP_LENGTH 32

static long long chunks_per_pass(const Options *options) {
    return options->last_chunk - options->first_chunk;
}

void report_progress_if_due(const Run *run, Progress *progress) {
    double now = now_in_seconds();
    if (now - progress->last_printed_at < PROGRESS_INTERVAL_SECONDS) {
        return;
    }
    progress->last_printed_at = now;
    long long chunks = chunks_per_pass(run->options);
    long long chunks_left = chunks - progress->chunks_done + (long long) (run->passes - progress->pass) * chunks;
    double chunks_per_second = progress->chunks_done / (now - progress->pass_started_at);
    printf("PROGRESS PASS=%u PASSES=%u CHUNK=%lld CHUNKS=%lld PHOTONS_PER_SECOND=%.6e ELAPSED=%.1f ETA=%.1f\n",
           progress->pass, run->passes, progress->chunks_done, chunks,
           chunks_per_second * run->options->chunk_size, now - run->started_at, chunks_left / chunks_per_second);
}

static void print_mu_bins(const Run *run, const Pass *pass) {
    printf("MU_BINS=");
    for (unsigned int slot = 0; slot < pass->observer_count; slot++) {
        const Observer *observer = &run->observers->observers[pass->first_observer + slot];
        printf("%u%s", observer->mu_bin, slot + 1 < pass->observer_count ? "," : " ");
    }
}

void print_pass(const Run *run, const Pass *pass, const PassResult *result) {
    const PhotonCounts *counts = &result->counts;
    printf("PASS PASS=%u PASSES=%u OBSERVERS=%u ", pass->number, run->passes, pass->observer_count);
    print_mu_bins(run, pass);
    printf("PHOTONS=%lld BLOCKED=%lld MISSED=%lld RECORDED=%lld ",
           counts->simulated, counts->blocked_by_spots, counts->missed_observers, counts->recorded);
    printf("ZERO_TIME=%.6f PHOTONS_TIME=%.6f COPY_TIME=%.6f SAVE_TIME=%.6f\n",
           result->times.zero, result->times.photons, result->times.copy, result->times.save);
}

static void write_processor(FILE *file, const ProcessorInfo *processor, const char *separator) {
    const ProcessorHardware *hardware = &processor->hardware;
    const KernelResources *kernel = &processor->kernel;
    const char *s = separator;
    fprintf(file, "PROCESSOR=%s%sPROCESSOR_NAME=%s%sCOMPUTE_CAPABILITY=%s%s", processor->kind, s, processor->name, s,
            processor->compute_capability, s);
    fprintf(file, "COMPUTE_UNITS=%u%sCOMPUTE_UNITS_TOTAL=%u%sUNIT=%s%s", processor->compute_units, s,
            processor->compute_units_total, s, processor->unit, s);
    fprintf(file, "MEMORY_TOTAL=%zu%sMEMORY_FREE=%zu%sMEMORY_FOR_IMAGE=%zu%sL2_BYTES=%zu%s", hardware->memory_total, s,
            hardware->memory_free, s, hardware->memory_for_image, s, hardware->l2_bytes, s);
    fprintf(file, "PEAK_BANDWIDTH=%.4g%sFP32_PER_FP64=%d%sMAX_THREADS_PER_UNIT=%d%s", hardware->peak_bandwidth, s,
            hardware->fp32_per_fp64, s, hardware->max_threads_per_unit, s);
    fprintf(file, "CUDA_RUNTIME=%d%sCUDA_DRIVER=%d%s", hardware->runtime_version, s, hardware->driver_version, s);
    fprintf(file, "REGISTERS_PER_THREAD=%d%sLOCAL_BYTES_PER_THREAD=%zu%sCONSTANT_BYTES=%zu%s",
            kernel->registers_per_thread, s, kernel->local_bytes_per_thread, s, kernel->constant_bytes, s);
    fprintf(file, "THREADS_PER_BLOCK=%d%sBLOCKS_PER_UNIT=%d%sBLOCKS=%d%sOCCUPANCY=%.3f%s", kernel->threads_per_block, s,
            kernel->blocks_per_unit, s, kernel->blocks, s, kernel->occupancy, s);
}

static void print_configuration(const Run *run) {
    const Options *options = run->options;
    const StarConfig *star = &options->star;
    write_processor(stdout, run->processor, " ");
    printf("HOST=%s PRECISION=%d N=%lld SEED=%llu A=%g SPOTS=%u UMBRA_RADIUS=%g PENUMBRA_RADIUS=%g ",
           run->host, PRECISION_BITS, star->photons, star->seed, star->limb_darkening, star->spots,
           star->umbra_radius, star->penumbra_radius);
    printf("SIZE=%u FRAMES=%u NMU=%u OBSERVERS=%u OBSERVERS_PER_PASS=%u PASSES=%u ",
           star->size, star->nphi, star->nmu, star->observer_count, options->observers_per_pass, run->passes);
    printf("CHUNK_SIZE=%lld CHUNKS=%lld FIRST_CHUNK=%lld LAST_CHUNK=%lld ",
           options->chunk_size, chunk_count(options), options->first_chunk, options->last_chunk);
}

static void print_observer(const Observer *observer, unsigned int observers_per_pass) {
    printf("OBSERVER OBSERVER=%u MU_BIN=%u INCLINATION_REQUESTED=%.10g INCLINATION_BIN_CENTER=%.10g PASS=%u FILE=%s\n",
           observer->index, observer->mu_bin, observer->requested_inclination, observer->bin_center_inclination,
           observer->index / observers_per_pass + 1, observer->file_stem);
}

void print_config(const Run *run) {
    const Options *options = run->options;
    size_t observer_bytes = pixels_per_observer(&options->star) * sizeof(PixelCount);
    printf("CONFIG ");
    print_configuration(run);
    printf("RECORD=%d OBSERVER_BYTES=%zu IMAGE_BYTES=%zu\n", options->record ? 1 : 0, observer_bytes,
           options->record ? observer_bytes * options->observers_per_pass : 0);
    for (unsigned int index = 0; index < run->observers->count; index++) {
        print_observer(&run->observers->observers[index], options->observers_per_pass);
    }
}

void print_summary(const Run *run, const PhotonCounts *counts, const RunTimes *times, size_t image_bytes) {
    const Options *options = run->options;
    double photons_traced = (double) counts->simulated * run->passes;
    printf("SUMMARY ");
    print_configuration(run);
    printf("PHOTONS=%lld BLOCKED=%lld MISSED=%lld RECORDED=%lld RECORD=%d IMAGE_BYTES=%zu ",
           counts->simulated, counts->blocked_by_spots, counts->missed_observers, counts->recorded,
           options->record ? 1 : 0, image_bytes);
    printf("SETUP_TIME=%.6f ZERO_TIME=%.6f PHOTONS_TIME=%.6f COPY_TIME=%.6f SAVE_TIME=%.6f TIME=%.6f "
           "PHOTONS_PER_SECOND=%.6e ",
           times->setup, times->zero, times->photons, times->copy, times->save, times->total,
           times->photons > 0 ? photons_traced / times->photons : 0.0);
    printf("OUTPUT_DIR=%s\n", options->record ? options->output_dir : "none");
}

static void format_utc_now(char *text) {
    time_t now = time(NULL);
    struct tm utc;
    gmtime_r(&now, &utc);
    strftime(text, TIMESTAMP_LENGTH, "%Y-%m-%dT%H:%M:%SZ", &utc);
}

static void write_record_lines(FILE *file, const Run *run, const Pass *pass, const Observer *observer,
                               const PassResult *pass_result, const ObserverResult *result) {
    const Options *options = run->options;
    const StarConfig *star = &options->star;
    char finished[TIMESTAMP_LENGTH];
    format_utc_now(finished);
    fprintf(file, "FINGERPRINT=%s\nCODE_HASH=%s\n", options->fingerprint, options->code_hash);
    fprintf(file, "IMAGE=%s.bin\nOBSERVER=%u\nMU_BIN=%u\nINCLINATION_REQUESTED=%.10g\nINCLINATION_BIN_CENTER=%.10g\n",
            observer->file_stem, observer->index, observer->mu_bin,
            observer->requested_inclination, observer->bin_center_inclination);
    write_processor(file, run->processor, "\n");
    fprintf(file, "HOST=%s\nPRECISION=%d\n", run->host, PRECISION_BITS);
    fprintf(file, "N=%lld\nSEED=%llu\nA=%g\nSPOTS=%u\nUMBRA_RADIUS=%g\nPENUMBRA_RADIUS=%g\n",
            star->photons, star->seed, star->limb_darkening, star->spots, star->umbra_radius, star->penumbra_radius);
    fprintf(file, "SIZE=%u\nFRAMES=%u\nNMU=%u\n", star->size, star->nphi, star->nmu);
    fprintf(file, "CHUNK_SIZE=%lld\nFIRST_CHUNK=%lld\nLAST_CHUNK=%lld\n",
            options->chunk_size, options->first_chunk, options->last_chunk);
    fprintf(file, "PASS=%u\nPASSES=%u\nPASS_OBSERVERS=%u\n", pass->number, run->passes, pass->observer_count);
    fprintf(file, "PHOTONS=%lld\nBLOCKED=%lld\nRECORDED=%llu\nIMAGE_BYTES=%zu\n",
            pass_result->counts.simulated, pass_result->counts.blocked_by_spots, result->recorded, result->bytes);
    fprintf(file, "PHOTONS_TIME=%.6f\nCOPY_TIME=%.6f\nSAVE_TIME=%.6f\nFINISHED=%s\n",
            pass_result->times.photons, result->copy_time, result->save_time, finished);
}

int write_observer_record(const Run *run, const Pass *pass, const Observer *observer,
                          const PassResult *pass_result, const ObserverResult *result) {
    char path[PATH_LENGTH], partial_path[PARTIAL_PATH_LENGTH];
    observer_path(run->options->output_dir, observer, ".record", path);
    partial_path_of(path, partial_path);
    FILE *file = fopen(partial_path, "w");
    if (file == NULL) {
        perror("Could not open the record file");
        return -1;
    }
    write_record_lines(file, run, pass, observer, pass_result, result);
    if (close_on_disk(file) != 0) {
        fprintf(stderr, "Could not write the record %s.\n", partial_path);
        return -1;
    }
    return move_into_place(partial_path, path);
}
