#ifndef GPU_SMS_CUH
#define GPU_SMS_CUH

#define MAX_SM_IDS 1024

typedef struct {
    int allowed[MAX_SM_IDS];
    unsigned long long chunks[MAX_SM_IDS];
    unsigned long long photons[MAX_SM_IDS];
    unsigned long long recorded[MAX_SM_IDS];
    unsigned long long cycles[MAX_SM_IDS];
} SmWork;

static inline __device__ unsigned int sm_id(void) {
    unsigned int id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

static inline __device__ bool is_allowed(unsigned int sm, const SmWork *sm_work) {
    return sm < MAX_SM_IDS && sm_work->allowed[sm];
}

static inline __device__ void add_sm_work(SmWork *sm_work, unsigned int sm, unsigned long long chunks,
                                          unsigned long long photons, unsigned long long recorded,
                                          unsigned long long cycles) {
    atomicAdd(&sm_work->chunks[sm], chunks);
    atomicAdd(&sm_work->photons[sm], photons);
    atomicAdd(&sm_work->recorded[sm], recorded);
    atomicMax(&sm_work->cycles[sm], cycles);
}

unsigned int resolve_sm_count(unsigned int requested);
SmWork *create_sm_work(unsigned int sms, int blocks, int threads_per_block);
void reset_sm_work(SmWork *sm_work);
void print_sm_work(const SmWork *sm_work, unsigned int pass);
void free_sm_work(SmWork *sm_work);

#endif
