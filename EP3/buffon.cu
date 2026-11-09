// EP3: Buffon's needle on the GPU (CUDA), compared with EP1 on the CPU.
//
//   ./buffon -n 1e9                  1e9 needles on the whole GPU, double precision
//   ./buffon -n 1e9 -m 4 -p 32       only 4 SMs, float precision
//   ./buffon -n 1e8 -c               one CPU core, as in EP1
//   ./buffon -i                      the GPU's properties
//   ./buffon -h                      help
//
// Build: nvcc buffon.cu -o buffon -O2 -arch=native

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

// LINK_TO_COLLAB="https://colab.research.google.com/drive/1k3lFfxoS_p-HhlmDtAQS_6qJdhhLcMSy#scrollTo=Vxug-cFT6ym5"
const double PI = M_PI;
const double L = 1.0;
const double D = 1.0;

#define MIN_CHUNK_SIZE 64
#define MAX_CHUNK_SIZE 4096
#define CHUNKS_PER_THREAD 4
#define MAX_SM_IDS 1024
#define MAX_THREADS_PER_BLOCK 1024

typedef curandStatePhilox4_32_10_t Rng;

typedef struct {
    long long n;
    unsigned int seed;
    int precision;
    int sms;
    bool cpu;
    bool info;
} Config;

typedef struct {
    int threads_per_block;
    int blocks_per_sm;
    int blocks;
    bool cooperative;
} Launch;

typedef struct {
    int *sm_allowed;
    int *sm_used;
    unsigned long long *next_chunk;
    unsigned long long *crossings;
    unsigned long long *cycles;
} GpuCounters;

typedef struct {
    const char *device;
    int sms;
    int sms_used;
    int blocks;
    int threads_per_block;
    long long chunk_size;
    unsigned long long crossings;
    double time;
    double loop_time;
    unsigned long long cycles;
    int clock_mhz;
} Result;


// ============================================================ command line

void print_usage(const char *program) {
    printf("Usage: %s -n N [-s seed] [-p 64|32|16] [-m sms] [-c]\n"
           "       %s -i\n\n"
           "  -n N          number of needles, e.g. 1000000 or 1e9 (required)\n"
           "  -s seed       random seed (default: current time)\n"
           "  -p precision  64 = double, 32 = float, 16 = half (default: 64)\n"
           "  -m sms        number of GPU SMs (\"cores\") to use (default: 0 = all)\n"
           "  -c            run on one CPU core, as in EP1 (drand48), instead of the GPU\n"
           "  -i            print the GPU's properties and exit\n"
           "  -h            show this help\n",
           program, program);
}

void stop(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

bool parse_number(const char *text, double *value) {
    char *end;
    *value = strtod(text, &end);
    return end != text && *end == '\0';
}

double number_or_stop(const char *text, double smallest, double largest, const char *message) {
    double value;
    if (!parse_number(text, &value) || value < smallest || value > largest) {
        stop(message);
    }
    return value;
}

int precision_or_stop(const char *text) {
    const char *message = "-p must be 64, 32 or 16.";
    int precision = (int) number_or_stop(text, 16, 64, message);
    if (precision != 64 && precision != 32 && precision != 16) {
        stop(message);
    }
    return precision;
}

Config default_config() {
    Config config;
    config.n = 0;
    config.seed = (unsigned int) time(NULL);
    config.precision = 64;
    config.sms = 0;
    config.cpu = false;
    config.info = false;
    return config;
}

void check_config(Config config, int argc, char *argv[]) {
    if (optind < argc) {
        fprintf(stderr, "Unexpected argument: %s\n", argv[optind]);
        exit(1);
    }
    if (!config.info && config.n == 0) {
        print_usage(argv[0]);
        stop("-n is required.");
    }
    if (config.cpu && config.sms != 0) {
        stop("-m chooses GPU SMs; it cannot be used with -c.");
    }
}

Config load_config(int argc, char *argv[]) {
    Config config = default_config();
    int flag;
    while ((flag = getopt(argc, argv, "n:s:p:m:cih")) != -1) {
        switch (flag) {
        case 'n':
            config.n = (long long) number_or_stop(optarg, 1, 9e18, "-n must be a number from 1 to 9e18.");
            break;
        case 's':
            config.seed = (unsigned int) number_or_stop(optarg, 0, 4294967295.0, "-s must be a number from 0 to 4294967295.");
            break;
        case 'p':
            config.precision = precision_or_stop(optarg);
            break;
        case 'm':
            config.sms = (int) number_or_stop(optarg, 0, MAX_SM_IDS, "-m must be 0 (all SMs) or a number of SMs.");
            break;
        case 'c':
            config.cpu = true;
            break;
        case 'i':
            config.info = true;
            break;
        case 'h':
            print_usage(argv[0]);
            exit(0);
        default:
            print_usage(argv[0]);
            exit(1);
        }
    }
    check_config(config, argc, argv);
    return config;
}


// ============================================================ helpers

double now_in_seconds() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

void check(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error while %s: %s\n", what, cudaGetErrorString(error));
        exit(1);
    }
}

Result empty_result(const char *device) {
    Result result;
    result.device = device;
    result.sms = 1;
    result.sms_used = 1;
    result.blocks = 1;
    result.threads_per_block = 1;
    result.chunk_size = 0;
    result.crossings = 0;
    result.time = 0;
    result.loop_time = 0;
    result.cycles = 0;
    result.clock_mhz = 0;
    return result;
}


// ============================================================ one needle on the CPU

double cpu_random_double() {
    return drand48();
}

float cpu_random_float() {
    return (float) drand48();
}

float round_to_half(float value) {
    return __half2float(__float2half(value));
}

bool cpu_needle_crosses_fp64() {
    double x = cpu_random_double() * (D/2.0);
    double theta = cpu_random_double() * (PI/2.0);
    return x <= (L/2.0) * sin(theta);
}

bool cpu_needle_crosses_fp32() {
    float x = cpu_random_float() * (float) (D/2.0);
    float theta = cpu_random_float() * (float) (PI/2.0);
    return x <= (float) (L/2.0) * sinf(theta);
}

bool cpu_needle_crosses_fp16() {
    float x = round_to_half(round_to_half(cpu_random_float()) * round_to_half((float) (D/2.0)));
    float theta = round_to_half(round_to_half(cpu_random_float()) * round_to_half((float) (PI/2.0)));
    float reach = round_to_half(round_to_half((float) (L/2.0)) * round_to_half(sinf(theta)));
    return x <= reach;
}

bool cpu_needle_crosses(int precision) {
    if (precision == 64) {
        return cpu_needle_crosses_fp64();
    }
    if (precision == 32) {
        return cpu_needle_crosses_fp32();
    }
    return cpu_needle_crosses_fp16();
}


// ============================================================ the experiment on the CPU (EP1)

unsigned long long count_crossings_on_cpu(long long n, int precision) {
    unsigned long long crossings = 0;
    for (long long i = 0; i < n; i++) {
        if (cpu_needle_crosses(precision)) {
            crossings++;
        }
    }
    return crossings;
}

Result run_on_cpu(Config config) {
    srand48(config.seed);
    Result result = empty_result("CPU");
    double started_at = now_in_seconds();
    result.crossings = count_crossings_on_cpu(config.n, config.precision);
    result.time = now_in_seconds() - started_at;
    result.loop_time = result.time;
    return result;
}


// ============================================================ one needle on the GPU

__device__ bool gpu_needle_crosses_fp64(Rng *rng) {
    double x = curand_uniform_double(rng) * (D/2.0);
    double theta = curand_uniform_double(rng) * (PI/2.0);
    return x <= (L/2.0) * sin(theta);
}

__device__ bool gpu_needle_crosses_fp32(Rng *rng) {
    float x = curand_uniform(rng) * (float) (D/2.0);
    float theta = curand_uniform(rng) * (float) (PI/2.0);
    return x <= (float) (L/2.0) * sinf(theta);
}

__device__ bool gpu_needle_crosses_fp16(Rng *rng) {
    __half x = __hmul(__float2half(curand_uniform(rng)), __float2half((float) (D/2.0)));
    __half theta = __hmul(__float2half(curand_uniform(rng)), __float2half((float) (PI/2.0)));
    __half reach = __hmul(__float2half((float) (L/2.0)), hsin(theta));
    return __hle(x, reach);
}

__device__ bool gpu_needle_crosses(int precision, Rng *rng) {
    if (precision == 64) {
        return gpu_needle_crosses_fp64(rng);
    }
    if (precision == 32) {
        return gpu_needle_crosses_fp32(rng);
    }
    return gpu_needle_crosses_fp16(rng);
}


// ============================================================ the kernels

__device__ unsigned int sm_id() {
    unsigned int id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

__device__ long long smaller(long long a, long long b) {
    return a < b ? a : b;
}

__device__ bool is_allowed(unsigned int sm, const int *sm_allowed) {
    return sm < MAX_SM_IDS && sm_allowed[sm];
}

__device__ unsigned long long count_crossings_in_chunk(long long chunk, long long chunk_size, long long n,
                                                       unsigned int seed, int precision) {
    Rng rng;
    curand_init(seed, chunk, 0, &rng);
    long long first = chunk * chunk_size;
    long long last = smaller(first + chunk_size, n);
    unsigned long long crossings = 0;
    for (long long i = first; i < last; i++) {
        if (gpu_needle_crosses(precision, &rng)) {
            crossings++;
        }
    }
    return crossings;
}

__device__ long long take_next_chunk(unsigned long long *next_chunk) {
    return (long long) atomicAdd(next_chunk, 1ULL);
}

__device__ unsigned long long take_chunks_until_none_left(long long n, unsigned int seed, int precision,
                                                          long long chunk_size, unsigned long long *next_chunk) {
    long long chunks = (n + chunk_size - 1) / chunk_size;
    unsigned long long crossings = 0;
    long long chunk = take_next_chunk(next_chunk);
    while (chunk < chunks) {
        crossings += count_crossings_in_chunk(chunk, chunk_size, n, seed, precision);
        chunk = take_next_chunk(next_chunk);
    }
    return crossings;
}

__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK) throw_needles(long long n, unsigned int seed, int precision,
                                                                       long long chunk_size, GpuCounters counters) {
    long long started_at = clock64();
    unsigned int sm = sm_id();
    if (!is_allowed(sm, counters.sm_allowed)) {
        return;
    }
    if (threadIdx.x == 0) {
        counters.sm_used[sm] = 1;
    }
    atomicAdd(counters.crossings, take_chunks_until_none_left(n, seed, precision, chunk_size, counters.next_chunk));
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(counters.cycles, (unsigned long long) (clock64() - started_at));
    }
}

__global__ void mark_sm_ids(int *seen) {
    unsigned int sm = sm_id();
    if (threadIdx.x == 0 && sm < MAX_SM_IDS) {
        seen[sm] = 1;
    }
}


// ============================================================ the GPU and the launch

cudaDeviceProp gpu_properties() {
    cudaDeviceProp gpu;
    check(cudaGetDeviceProperties(&gpu, 0), "reading the GPU properties");
    return gpu;
}

int gpu_attribute(cudaDeviceAttr attribute) {
    int value = 0;
    check(cudaDeviceGetAttribute(&value, attribute, 0), "reading a GPU attribute");
    return value;
}

int gpu_clock_mhz() {
    return gpu_attribute(cudaDevAttrClockRate) / 1000;
}

int gpu_sm_count() {
    return gpu_attribute(cudaDevAttrMultiProcessorCount);
}

Launch choose_launch() {
    Launch launch;
    int smallest_full_grid;
    check(cudaOccupancyMaxPotentialBlockSize(&smallest_full_grid, &launch.threads_per_block, throw_needles, 0, 0),
          "choosing the block size");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&launch.blocks_per_sm, throw_needles, launch.threads_per_block, 0),
          "counting blocks per SM");
    launch.blocks = launch.blocks_per_sm * gpu_sm_count();
    launch.cooperative = gpu_attribute(cudaDevAttrCooperativeLaunch) == 1;
    return launch;
}

long long total_threads(Launch launch) {
    return (long long) launch.blocks * launch.threads_per_block;
}

long long choose_chunk_size(long long n, Launch launch) {
    long long size = n / (CHUNKS_PER_THREAD * total_threads(launch));
    if (size < MIN_CHUNK_SIZE) {
        return MIN_CHUNK_SIZE;
    }
    if (size > MAX_CHUNK_SIZE) {
        return MAX_CHUNK_SIZE;
    }
    return size;
}

void run_kernel(const void *kernel, Launch launch, void **arguments) {
    if (launch.cooperative) {
        check(cudaLaunchCooperativeKernel(kernel, launch.blocks, launch.threads_per_block, arguments, 0, 0),
              "launching the kernel");
    } else {
        check(cudaLaunchKernel(kernel, launch.blocks, launch.threads_per_block, arguments, 0, 0),
              "launching the kernel");
    }
    check(cudaDeviceSynchronize(), "waiting for the kernel");
}


// ============================================================ GPU memory

void *gpu_zeroed_memory(size_t bytes) {
    void *memory;
    check(cudaMalloc(&memory, bytes), "allocating GPU memory");
    check(cudaMemset(memory, 0, bytes), "zeroing GPU memory");
    return memory;
}

void copy_to_gpu(void *gpu_memory, const void *memory, size_t bytes) {
    check(cudaMemcpy(gpu_memory, memory, bytes, cudaMemcpyHostToDevice), "copying to the GPU");
}

void copy_from_gpu(void *memory, const void *gpu_memory, size_t bytes) {
    check(cudaMemcpy(memory, gpu_memory, bytes, cudaMemcpyDeviceToHost), "copying from the GPU");
}

void free_gpu_memory(void *gpu_memory) {
    check(cudaFree(gpu_memory), "freeing GPU memory");
}

GpuCounters create_gpu_counters(const int *sm_allowed) {
    GpuCounters counters;
    counters.sm_allowed = (int *) gpu_zeroed_memory(MAX_SM_IDS * sizeof(int));
    counters.sm_used = (int *) gpu_zeroed_memory(MAX_SM_IDS * sizeof(int));
    counters.next_chunk = (unsigned long long *) gpu_zeroed_memory(sizeof(unsigned long long));
    counters.crossings = (unsigned long long *) gpu_zeroed_memory(sizeof(unsigned long long));
    counters.cycles = (unsigned long long *) gpu_zeroed_memory(sizeof(unsigned long long));
    copy_to_gpu(counters.sm_allowed, sm_allowed, MAX_SM_IDS * sizeof(int));
    return counters;
}

void free_gpu_counters(GpuCounters counters) {
    free_gpu_memory(counters.sm_allowed);
    free_gpu_memory(counters.sm_used);
    free_gpu_memory(counters.next_chunk);
    free_gpu_memory(counters.crossings);
    free_gpu_memory(counters.cycles);
}

unsigned long long read_gpu_counter(const unsigned long long *gpu_counter) {
    unsigned long long value;
    copy_from_gpu(&value, gpu_counter, sizeof(value));
    return value;
}

int count_marked(const int *marks) {
    int count = 0;
    for (int id = 0; id < MAX_SM_IDS; id++) {
        count += marks[id];
    }
    return count;
}

int count_sms_used(const int *gpu_sm_used) {
    int sm_used[MAX_SM_IDS];
    copy_from_gpu(sm_used, gpu_sm_used, sizeof(sm_used));
    return count_marked(sm_used);
}


// ============================================================ choosing the SMs

int list_sm_ids(Launch launch, int *sm_ids) {
    int *gpu_seen = (int *) gpu_zeroed_memory(MAX_SM_IDS * sizeof(int));
    void *arguments[] = {&gpu_seen};
    run_kernel((const void *) mark_sm_ids, launch, arguments);
    int seen[MAX_SM_IDS];
    copy_from_gpu(seen, gpu_seen, sizeof(seen));
    free_gpu_memory(gpu_seen);
    int found = 0;
    for (int id = 0; id < MAX_SM_IDS; id++) {
        if (seen[id]) {
            sm_ids[found] = id;
            found++;
        }
    }
    return found;
}

int resolve_sm_count(int requested) {
    int available = gpu_sm_count();
    if (requested > available) {
        fprintf(stderr, "This GPU has only %d SMs.\n", available);
        exit(1);
    }
    return requested == 0 ? available : requested;
}

void allow_first_sms(int sms, Launch launch, int *sm_allowed) {
    int sm_ids[MAX_SM_IDS];
    int found = list_sm_ids(launch, sm_ids);
    if (sms > found) {
        stop("The GPU has fewer working SMs than requested.");
    }
    memset(sm_allowed, 0, MAX_SM_IDS * sizeof(int));
    for (int i = 0; i < sms; i++) {
        sm_allowed[sm_ids[i]] = 1;
    }
}


// ============================================================ the experiment on the GPU

double time_needle_loop(Config config, Launch launch, long long chunk_size, GpuCounters counters) {
    void *arguments[] = {&config.n, &config.seed, &config.precision, &chunk_size, &counters};
    double started_at = now_in_seconds();
    run_kernel((const void *) throw_needles, launch, arguments);
    return now_in_seconds() - started_at;
}

Result run_on_gpu(Config config) {
    double started_at = now_in_seconds();
    int sms = resolve_sm_count(config.sms);
    Launch launch = choose_launch();
    long long chunk_size = choose_chunk_size(config.n, launch);
    int sm_allowed[MAX_SM_IDS];
    allow_first_sms(sms, launch, sm_allowed);
    GpuCounters counters = create_gpu_counters(sm_allowed);

    Result result = empty_result("GPU");
    result.loop_time = time_needle_loop(config, launch, chunk_size, counters);
    result.crossings = read_gpu_counter(counters.crossings);
    result.cycles = read_gpu_counter(counters.cycles);
    result.sms_used = count_sms_used(counters.sm_used);
    free_gpu_counters(counters);

    result.sms = sms;
    result.blocks = launch.blocks;
    result.threads_per_block = launch.threads_per_block;
    result.chunk_size = chunk_size;
    result.clock_mhz = gpu_clock_mhz();
    result.time = now_in_seconds() - started_at;
    return result;
}


// ============================================================ the result

double estimate_pi(long long n, unsigned long long crossings) {
    return (2.0*L*n)/(D*crossings);
}

double estimate_sigma(long long n, unsigned long long crossings) {
    double fraction = (double) crossings / n;
    return estimate_pi(n, crossings) * sqrt((1.0 - fraction) / (n * fraction));
}

void print_result(Config config, Result result) {
    double pi = estimate_pi(config.n, result.crossings);
    printf("DEVICE=%s PRECISION=%d N=%lld SEED=%u ", result.device, config.precision, config.n, config.seed);
    printf("SMS=%d SMS_USED=%d BLOCKS=%d THREADS=%d CHUNK_SIZE=%lld ",
           result.sms, result.sms_used, result.blocks, result.threads_per_block, result.chunk_size);
    printf("PI=%.15f ERROR=%.15e SIGMA=%.15e ", pi, fabs(pi - PI), estimate_sigma(config.n, result.crossings));
    printf("TIME=%.6f LOOP_TIME=%.6f CYCLES=%llu CLOCK_MHZ=%d\n",
           result.time, result.loop_time, result.cycles, result.clock_mhz);
}


// ============================================================ GPU info

int cuda_cores_per_sm(int major, int minor) {
    if (major == 7) {
        return 64;
    }
    if (major == 8 && minor == 0) {
        return 64;
    }
    if (major >= 8) {
        return 128;
    }
    return 0;
}

void replace_spaces_with_underscores(char *text) {
    for (char *c = text; *c; c++) {
        if (*c == ' ') {
            *c = '_';
        }
    }
}

int registers_per_thread() {
    cudaFuncAttributes kernel;
    check(cudaFuncGetAttributes(&kernel, throw_needles), "reading the kernel's properties");
    return kernel.numRegs;
}

void print_gpu_hardware(cudaDeviceProp gpu) {
    char name[256];
    snprintf(name, sizeof(name), "%s", gpu.name);
    replace_spaces_with_underscores(name);
    printf("NAME=%s COMPUTE_CAPABILITY=%d.%d SMS=%d CUDA_CORES_PER_SM=%d FP32_PER_FP64=%d WARP_SIZE=%d ",
           name, gpu.major, gpu.minor, gpu.multiProcessorCount, cuda_cores_per_sm(gpu.major, gpu.minor),
           gpu_attribute(cudaDevAttrSingleToDoublePrecisionPerfRatio), gpu.warpSize);
    printf("CLOCK_MHZ=%d MEMORY_GB=%.1f ", gpu_clock_mhz(), gpu.totalGlobalMem / 1e9);
}

void print_gpu_limits(cudaDeviceProp gpu) {
    printf("MAX_THREADS_PER_BLOCK=%d MAX_THREADS_PER_SM=%d MAX_BLOCKS_PER_SM=%d REGISTERS_PER_SM=%d ",
           gpu.maxThreadsPerBlock, gpu.maxThreadsPerMultiProcessor, gpu.maxBlocksPerMultiProcessor,
           gpu.regsPerMultiprocessor);
}

void print_launch(Launch launch) {
    int sm_ids[MAX_SM_IDS];
    printf("COOPERATIVE_LAUNCH=%d THREADS_PER_BLOCK=%d BLOCKS_PER_SM=%d REGISTERS_PER_THREAD=%d SM_IDS_FOUND=%d",
           launch.cooperative, launch.threads_per_block, launch.blocks_per_sm, registers_per_thread(),
           list_sm_ids(launch, sm_ids));
}

void print_gpu_info() {
    cudaDeviceProp gpu = gpu_properties();
    print_gpu_hardware(gpu);
    print_gpu_limits(gpu);
    print_launch(choose_launch());
    printf("\n");
}


// ============================================================ main

void check_crossings(Result result) {
    if (result.crossings == 0) {
        stop("No needle crossed a line, so pi cannot be estimated: use a larger N.");
    }
}

int main(int argc, char *argv[]) {
    Config config = load_config(argc, argv);
    if (config.info) {
        print_gpu_info();
        return 0;
    }
    Result result = config.cpu ? run_on_cpu(config) : run_on_gpu(config);
    check_crossings(result);
    print_result(config, result);
    return 0;
}
