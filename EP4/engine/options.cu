#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <getopt.h>

#include "options.cuh"
#include "image.cuh"

#define DEFAULT_CHUNK_SIZE 1024
#define DEFAULT_OBSERVERS "1.5708,1.0472,0.5236,0"
#define ALL_CHUNKS (-1)
#define DERIVED_FROM_FRAMES 0
#define ALL_OBSERVERS 0
#define NO_BUDGET 0
#define TYPED_PI 3.1416

enum {
    OPT_SPOTS = 1000, OPT_UMBRA_RADIUS, OPT_PENUMBRA_RADIUS, OPT_OBSERVERS, OPT_SIZE, OPT_FRAMES, OPT_NMU,
    OPT_CHUNK_SIZE, OPT_FIRST_CHUNK, OPT_LAST_CHUNK, OPT_OBSERVERS_PER_PASS, OPT_MEMORY_BUDGET,
    OPT_SMS, OPT_NO_RECORD, OPT_FINGERPRINT, OPT_CODE_HASH, OPT_DESCRIBE, OPT_TEST_RNG
};

static const struct option LONG_OPTIONS[] = {
    {"photons", required_argument, 0, 'n'},
    {"seed", required_argument, 0, 's'},
    {"limb", required_argument, 0, 'a'},
    {"spots", required_argument, 0, OPT_SPOTS},
    {"umbra-radius", required_argument, 0, OPT_UMBRA_RADIUS},
    {"penumbra-radius", required_argument, 0, OPT_PENUMBRA_RADIUS},
    {"observers", required_argument, 0, OPT_OBSERVERS},
    {"size", required_argument, 0, OPT_SIZE},
    {"frames", required_argument, 0, OPT_FRAMES},
    {"nmu", required_argument, 0, OPT_NMU},
    {"chunk-size", required_argument, 0, OPT_CHUNK_SIZE},
    {"first-chunk", required_argument, 0, OPT_FIRST_CHUNK},
    {"last-chunk", required_argument, 0, OPT_LAST_CHUNK},
    {"observers-per-pass", required_argument, 0, OPT_OBSERVERS_PER_PASS},
    {"memory-budget", required_argument, 0, OPT_MEMORY_BUDGET},
    {"sms", required_argument, 0, OPT_SMS},
    {"no-record", no_argument, 0, OPT_NO_RECORD},
    {"output-dir", required_argument, 0, 'o'},
    {"fingerprint", required_argument, 0, OPT_FINGERPRINT},
    {"code-hash", required_argument, 0, OPT_CODE_HASH},
    {"describe", no_argument, 0, OPT_DESCRIBE},
    {"test-rng", no_argument, 0, OPT_TEST_RNG},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static void print_usage(const char *program) {
    printf("Usage: %s -n N [options]\n"
           "       %s --test-rng\n\n"
           "  -n, --photons N              number of photons, e.g. 1000000 or 1e9 (required)\n"
           "  -s, --seed S                 random seed (default 42)\n"
           "  -a, --limb A                 limb darkening a in I = I0 (1 + a mu) (default 1)\n"
           "      --spots K                number of spots (default 3)\n"
           "      --umbra-radius R         dark core radius, rad (default 0.05)\n"
           "      --penumbra-radius R      penumbra outer radius, rad (default 0.10)\n"
           "      --observers I1,I2,...    inclinations in radians, 0 to pi (default %s)\n"
           "      --size S                 square image of S x S pixels (default 200)\n"
           "      --frames F               azimuth bins = video frames (default 150)\n"
           "      --nmu M                  mu bins (default frames / pi)\n"
           "      --chunk-size C           photons per chunk (default %d)\n"
           "      --first-chunk A          first chunk to simulate (default 0)\n"
           "      --last-chunk B           one past the last chunk (default: all)\n"
           "      --observers-per-pass K   observers recorded per pass (default: all, one pass)\n"
           "      --memory-budget B        or K from a budget for the image: 7GB, 7G, 7e9, 512MB, 6.5GiB\n"
           "      --sms N                  the GPU engines: work on N SMs only (default: all)\n"
           "      --no-record              simulate the photons without an image\n"
           "  -o, --output-dir DIR         where the observer files go (default .)\n"
           "      --fingerprint F          the job's fingerprint, copied into the records\n"
           "      --code-hash H            the engine's code hash, copied into the records\n"
           "      --describe               print the derived configuration (nmu, bins, passes) and exit\n"
           "      --test-rng               check Philox against the Random123 test vectors\n"
           "  -h, --help                   this help\n",
           program, program, DEFAULT_OBSERVERS, DEFAULT_CHUNK_SIZE);
}

static void stop(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static double number_or_stop(const char *text, double smallest, double largest, const char *message) {
    char *end;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || value < smallest || value > largest) {
        stop(message);
    }
    return value;
}

static const char *word_or_stop(const char *text, const char *message) {
    if (*text == '\0' || strpbrk(text, " \t\n") != NULL) {
        stop(message);
    }
    return text;
}

static double bytes_per_unit(const char *unit) {
    static const struct { const char *name; double bytes; } UNITS[] = {
        {"", 1}, {"B", 1}, {"K", 1e3}, {"KB", 1e3}, {"M", 1e6}, {"MB", 1e6},
        {"G", 1e9}, {"GB", 1e9}, {"T", 1e12}, {"TB", 1e12},
        {"KiB", 1024.0}, {"MiB", 1048576.0}, {"GiB", 1073741824.0}, {"TiB", 1099511627776.0},
    };
    for (size_t i = 0; i < sizeof(UNITS) / sizeof(UNITS[0]); i++) {
        if (strcasecmp(unit, UNITS[i].name) == 0) {
            return UNITS[i].bytes;
        }
    }
    return 0;
}

static double bytes_or_stop(const char *text) {
    char *unit;
    double value = strtod(text, &unit);
    double bytes = value * bytes_per_unit(unit);
    if (unit == text || bytes <= 0) {
        stop("--memory-budget must be a size such as 7GB, 7G, 7e9, 512MB or 6.5GiB.");
    }
    return bytes;
}

static void read_inclinations(const char *text, StarConfig *star) {
    star->observer_count = 0;
    const char *cursor = text;
    while (true) {
        char *end;
        double inclination = strtod(cursor, &end);
        if (end == cursor || inclination < 0 || inclination > TYPED_PI || (*end != ',' && *end != '\0')) {
            stop("--observers must be inclinations in radians, from 0 to pi, separated by commas.");
        }
        if (star->observer_count == MAX_OBSERVERS) {
            stop("Too many observers.");
        }
        star->inclinations[star->observer_count++] = fmin(inclination, M_PI);
        if (*end == '\0') {
            return;
        }
        cursor = end + 1;
    }
}

static Options default_options(void) {
    Options options;
    options.star.photons = 0;
    options.star.seed = 42;
    options.star.limb_darkening = 1.0;
    options.star.spots = 3;
    options.star.umbra_radius = 0.05;
    options.star.penumbra_radius = 0.10;
    read_inclinations(DEFAULT_OBSERVERS, &options.star);
    options.star.size = 200;
    options.star.nphi = 150;
    options.star.nmu = DERIVED_FROM_FRAMES;
    options.chunk_size = DEFAULT_CHUNK_SIZE;
    options.first_chunk = 0;
    options.last_chunk = ALL_CHUNKS;
    options.observers_per_pass = ALL_OBSERVERS;
    options.memory_budget = NO_BUDGET;
    options.sms = ALL_SMS;
    options.record = true;
    options.output_dir = ".";
    options.fingerprint = "none";
    options.code_hash = "none";
    options.describe = false;
    options.test_rng = false;
    return options;
}

static void read_option(Options *options, int flag, const char *value, const char *program) {
    StarConfig *star = &options->star;
    switch (flag) {
    case 'n': star->photons = (long long) number_or_stop(value, 1, 9e18, "-n must be from 1 to 9e18."); break;
    case 's': star->seed = (unsigned long long) number_or_stop(value, 0, 9e15, "-s must be from 0 to 9e15."); break;
    case 'a': star->limb_darkening = number_or_stop(value, 0, 1e6, "-a must be 0 or more."); break;
    case OPT_SPOTS: star->spots = (unsigned int) number_or_stop(value, 0, MAX_SPOTS, "--spots must be from 0 to 64."); break;
    case OPT_UMBRA_RADIUS: star->umbra_radius = number_or_stop(value, 0, 3.14159, "--umbra-radius must be from 0 to pi."); break;
    case OPT_PENUMBRA_RADIUS: star->penumbra_radius = number_or_stop(value, 0, 3.14159, "--penumbra-radius must be from 0 to pi."); break;
    case OPT_OBSERVERS: read_inclinations(value, star); break;
    case OPT_SIZE: star->size = (unsigned int) number_or_stop(value, 1, 65536, "--size must be from 1 to 65536."); break;
    case OPT_FRAMES: star->nphi = (unsigned int) number_or_stop(value, 1, 65536, "--frames must be from 1 to 65536."); break;
    case OPT_NMU: star->nmu = (unsigned int) number_or_stop(value, 1, MAX_NMU, "--nmu must be from 1 to 4096."); break;
    case OPT_CHUNK_SIZE: options->chunk_size = (long long) number_or_stop(value, 1, 1e12, "--chunk-size must be 1 or more."); break;
    case OPT_FIRST_CHUNK: options->first_chunk = (long long) number_or_stop(value, 0, 9e18, "--first-chunk must be 0 or more."); break;
    case OPT_LAST_CHUNK: options->last_chunk = (long long) number_or_stop(value, 0, 9e18, "--last-chunk must be 0 or more."); break;
    case OPT_OBSERVERS_PER_PASS: options->observers_per_pass = (unsigned int) number_or_stop(value, 1, MAX_OBSERVERS, "--observers-per-pass must be 1 or more."); break;
    case OPT_MEMORY_BUDGET: options->memory_budget = bytes_or_stop(value); break;
    case OPT_SMS: options->sms = (unsigned int) number_or_stop(value, 1, 1024, "--sms must be from 1 to the GPU's SMs."); break;
    case OPT_NO_RECORD: options->record = false; break;
    case 'o': options->output_dir = word_or_stop(value, "--output-dir must not be empty or contain spaces."); break;
    case OPT_FINGERPRINT: options->fingerprint = word_or_stop(value, "--fingerprint must not be empty or contain spaces."); break;
    case OPT_CODE_HASH: options->code_hash = word_or_stop(value, "--code-hash must not be empty or contain spaces."); break;
    case OPT_DESCRIBE: options->describe = true; break;
    case OPT_TEST_RNG: options->test_rng = true; break;
    case 'h': print_usage(program); exit(0);
    default: print_usage(program); exit(1);
    }
}

static unsigned int nearest_odd(double value) {
    long odd = 2 * lround((value - 1) / 2) + 1;
    return odd < 1 ? 1 : (unsigned int) odd;
}

static unsigned int nmu_from_frames(unsigned int frames) {
    unsigned int nmu = nearest_odd(frames / M_PI);
    return nmu < MAX_NMU ? nmu : MAX_NMU - 1;
}

static unsigned int observers_within_budget(const Options *options) {
    double footprint = (double) pixels_per_observer(&options->star) * sizeof(PixelCount);
    unsigned int fitting = (unsigned int) fmin(options->star.observer_count, floor(options->memory_budget / footprint));
    if (fitting == 0) {
        fprintf(stderr, "One observer needs %.3g GB, more than the memory budget of %.3g GB.\n",
                footprint / 1e9, options->memory_budget / 1e9);
        exit(1);
    }
    return fitting;
}

static unsigned int balanced(unsigned int observers_per_pass, unsigned int observers) {
    unsigned int passes = (observers + observers_per_pass - 1) / observers_per_pass;
    return (observers + passes - 1) / passes;
}

static unsigned int largest_observers_per_pass(const Options *options) {
    unsigned int all = options->star.observer_count;
    if (options->observers_per_pass != ALL_OBSERVERS && options->memory_budget != NO_BUDGET) {
        stop("Use --observers-per-pass or --memory-budget, not both.");
    }
    if (!options->record) {
        return all;
    }
    if (options->memory_budget != NO_BUDGET) {
        return observers_within_budget(options);
    }
    if (options->observers_per_pass == ALL_OBSERVERS || options->observers_per_pass > all) {
        return all;
    }
    return options->observers_per_pass;
}

static unsigned int choose_observers_per_pass(const Options *options) {
    return balanced(largest_observers_per_pass(options), options->star.observer_count);
}

static void validate_options(Options *options, int argc, char *argv[]) {
    if (optind < argc) {
        fprintf(stderr, "Unexpected argument: %s\n", argv[optind]);
        exit(1);
    }
    if (options->test_rng) {
        return;
    }
    if (options->star.photons == 0) {
        print_usage(argv[0]);
        stop("-n is required.");
    }
    if (options->star.nmu == DERIVED_FROM_FRAMES) {
        options->star.nmu = nmu_from_frames(options->star.nphi);
    }
    long long chunks = chunk_count(options);
    if (options->last_chunk == ALL_CHUNKS) {
        options->last_chunk = chunks;
    }
    if (options->last_chunk > chunks || options->first_chunk > options->last_chunk) {
        fprintf(stderr, "The chunk range must be inside [0, %lld].\n", chunks);
        exit(1);
    }
    options->observers_per_pass = choose_observers_per_pass(options);
}

long long chunk_count(const Options *options) {
    return (options->star.photons + options->chunk_size - 1) / options->chunk_size;
}

unsigned int pass_count(const Options *options) {
    return (options->star.observer_count + options->observers_per_pass - 1) / options->observers_per_pass;
}

size_t pixels_per_observer(const StarConfig *star) {
    return (size_t) star->size * star->size * star->nphi;
}

Options load_options(int argc, char *argv[]) {
    Options options = default_options();
    int flag;
    while ((flag = getopt_long(argc, argv, "n:s:a:o:h", LONG_OPTIONS, NULL)) != -1) {
        read_option(&options, flag, optarg, argv[0]);
    }
    validate_options(&options, argc, argv);
    return options;
}
