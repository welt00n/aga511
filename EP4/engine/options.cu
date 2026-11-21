#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "options.cuh"

#define DEFAULT_CHUNK_SIZE 1024
#define ALL_CHUNKS (-1)

enum {
    OPT_SPOTS = 1000, OPT_UMBRA_RADIUS, OPT_PENUMBRA_RADIUS, OPT_OBSERVERS, OPT_NX, OPT_NY, OPT_NPHI, OPT_NMU,
    OPT_CHUNK_SIZE, OPT_FIRST_CHUNK, OPT_LAST_CHUNK, OPT_NO_RECORD, OPT_TEST_RNG
};

static const struct option LONG_OPTIONS[] = {
    {"photons", required_argument, 0, 'n'},
    {"seed", required_argument, 0, 's'},
    {"limb", required_argument, 0, 'a'},
    {"spots", required_argument, 0, OPT_SPOTS},
    {"umbra-radius", required_argument, 0, OPT_UMBRA_RADIUS},
    {"penumbra-radius", required_argument, 0, OPT_PENUMBRA_RADIUS},
    {"observers", required_argument, 0, OPT_OBSERVERS},
    {"nx", required_argument, 0, OPT_NX},
    {"ny", required_argument, 0, OPT_NY},
    {"nphi", required_argument, 0, OPT_NPHI},
    {"nmu", required_argument, 0, OPT_NMU},
    {"chunk-size", required_argument, 0, OPT_CHUNK_SIZE},
    {"first-chunk", required_argument, 0, OPT_FIRST_CHUNK},
    {"last-chunk", required_argument, 0, OPT_LAST_CHUNK},
    {"no-record", no_argument, 0, OPT_NO_RECORD},
    {"output", required_argument, 0, 'o'},
    {"test-rng", no_argument, 0, OPT_TEST_RNG},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static void print_usage(const char *program) {
    printf("Usage: %s -n N [options]\n"
           "       %s --test-rng\n\n"
           "  -n, --photons N           number of photons, e.g. 1000000 or 1e9 (required)\n"
           "  -s, --seed S              random seed (default 42)\n"
           "  -a, --limb A              limb darkening a in I = I0 (1 + a mu) (default 1)\n"
           "      --spots K             number of spots (default 3)\n"
           "      --umbra-radius R      dark core radius, rad (default 0.05)\n"
           "      --penumbra-radius R   penumbra outer radius, rad (default 0.10)\n"
           "      --observers K         observer latitudes (default 4)\n"
           "      --nx N, --ny N        image resolution (default 200 x 200)\n"
           "      --nphi N              azimuth bins = video frames (default 150)\n"
           "      --nmu N               mu bins (default 150)\n"
           "      --chunk-size C        photons per chunk (default %d)\n"
           "      --first-chunk A       first chunk to simulate (default 0)\n"
           "      --last-chunk B        one past the last chunk (default: all)\n"
           "      --no-record           simulate the photons without writing the image\n"
           "  -o, --output FILE         image file (default star.bin)\n"
           "      --test-rng            check Philox against the Random123 test vectors\n"
           "  -h, --help                this help\n",
           program, program, DEFAULT_CHUNK_SIZE);
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

static Options default_options(void) {
    Options options;
    options.star.photons = 0;
    options.star.seed = 42;
    options.star.limb_darkening = 1.0;
    options.star.spots = 3;
    options.star.umbra_radius = 0.05;
    options.star.penumbra_radius = 0.10;
    options.star.observers = 4;
    options.star.nx = 200;
    options.star.ny = 200;
    options.star.nphi = 150;
    options.star.nmu = 150;
    options.chunk_size = DEFAULT_CHUNK_SIZE;
    options.first_chunk = 0;
    options.last_chunk = ALL_CHUNKS;
    options.record = true;
    options.output = "star.bin";
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
    case OPT_OBSERVERS: star->observers = (unsigned int) number_or_stop(value, 1, MAX_NMU, "--observers must be from 1 to nmu."); break;
    case OPT_NX: star->nx = (unsigned int) number_or_stop(value, 1, 65536, "--nx must be from 1 to 65536."); break;
    case OPT_NY: star->ny = (unsigned int) number_or_stop(value, 1, 65536, "--ny must be from 1 to 65536."); break;
    case OPT_NPHI: star->nphi = (unsigned int) number_or_stop(value, 1, 65536, "--nphi must be from 1 to 65536."); break;
    case OPT_NMU: star->nmu = (unsigned int) number_or_stop(value, 1, MAX_NMU, "--nmu must be from 1 to 4096."); break;
    case OPT_CHUNK_SIZE: options->chunk_size = (long long) number_or_stop(value, 1, 1e12, "--chunk-size must be 1 or more."); break;
    case OPT_FIRST_CHUNK: options->first_chunk = (long long) number_or_stop(value, 0, 9e18, "--first-chunk must be 0 or more."); break;
    case OPT_LAST_CHUNK: options->last_chunk = (long long) number_or_stop(value, 0, 9e18, "--last-chunk must be 0 or more."); break;
    case OPT_NO_RECORD: options->record = false; break;
    case 'o': options->output = value; break;
    case OPT_TEST_RNG: options->test_rng = true; break;
    case 'h': print_usage(program); exit(0);
    default: print_usage(program); exit(1);
    }
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
    long long chunks = chunk_count(options);
    if (options->last_chunk == ALL_CHUNKS) {
        options->last_chunk = chunks;
    }
    if (options->last_chunk > chunks || options->first_chunk > options->last_chunk) {
        fprintf(stderr, "The chunk range must be inside [0, %lld].\n", chunks);
        exit(1);
    }
}

long long chunk_count(const Options *options) {
    return (options->star.photons + options->chunk_size - 1) / options->chunk_size;
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
