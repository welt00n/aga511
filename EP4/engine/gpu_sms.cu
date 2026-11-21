#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "gpu_sms.cuh"
#include "gpu_device.cuh"
#include "options.cuh"

__global__ void mark_sm_ids(int *seen) {
    unsigned int sm = sm_id();
    if (threadIdx.x == 0 && sm < MAX_SM_IDS) {
        seen[sm] = 1;
    }
}

static int list_sm_ids(int blocks, int threads_per_block, int *sm_ids) {
    int *gpu_seen;
    check(cudaMalloc(&gpu_seen, MAX_SM_IDS * sizeof(int)), "allocating the SM list");
    check(cudaMemset(gpu_seen, 0, MAX_SM_IDS * sizeof(int)), "zeroing the SM list");
    mark_sm_ids<<<blocks, threads_per_block>>>(gpu_seen);
    check(cudaGetLastError(), "launching mark_sm_ids");
    int seen[MAX_SM_IDS];
    check(cudaMemcpy(seen, gpu_seen, sizeof(seen), cudaMemcpyDeviceToHost), "reading the SM list");
    cudaFree(gpu_seen);
    int found = 0;
    for (int id = 0; id < MAX_SM_IDS; id++) {
        if (seen[id]) {
            sm_ids[found++] = id;
        }
    }
    return found;
}

unsigned int resolve_sm_count(unsigned int requested) {
    unsigned int available = (unsigned int) gpu_attribute(cudaDevAttrMultiProcessorCount);
    if (requested > available) {
        fprintf(stderr, "This GPU has only %u SMs.\n", available);
        exit(1);
    }
    return requested == ALL_SMS ? available : requested;
}

static void allow_first_sms(unsigned int sms, int blocks, int threads_per_block, int *allowed) {
    int sm_ids[MAX_SM_IDS];
    int found = list_sm_ids(blocks, threads_per_block, sm_ids);
    if ((int) sms > found) {
        fprintf(stderr, "Only %d SMs answered, fewer than the %u requested.\n", found, sms);
        exit(1);
    }
    for (unsigned int i = 0; i < sms; i++) {
        allowed[sm_ids[i]] = 1;
    }
}

SmWork *create_sm_work(unsigned int sms, int blocks, int threads_per_block) {
    SmWork *host_work = (SmWork *) calloc(1, sizeof(SmWork));
    allow_first_sms(sms, blocks, threads_per_block, host_work->allowed);
    SmWork *sm_work;
    check(cudaMalloc(&sm_work, sizeof(SmWork)), "allocating the work per SM");
    check(cudaMemcpy(sm_work, host_work, sizeof(SmWork), cudaMemcpyHostToDevice), "copying the allowed SMs");
    free(host_work);
    return sm_work;
}

void reset_sm_work(SmWork *sm_work) {
    size_t counters = sizeof(SmWork) - offsetof(SmWork, chunks);
    check(cudaMemset((char *) sm_work + offsetof(SmWork, chunks), 0, counters), "resetting the work per SM");
}

void print_sm_work(const SmWork *sm_work, unsigned int pass) {
    SmWork *work = (SmWork *) malloc(sizeof(SmWork));
    check(cudaMemcpy(work, sm_work, sizeof(SmWork), cudaMemcpyDeviceToHost), "reading the work per SM");
    unsigned int used = 0;
    unsigned long long slowest = 0;
    for (unsigned int sm = 0; sm < MAX_SM_IDS; sm++) {
        if (work->chunks[sm] == 0) {
            continue;
        }
        used++;
        slowest = work->cycles[sm] > slowest ? work->cycles[sm] : slowest;
        printf("SM PASS=%u SM=%u CHUNKS=%llu PHOTONS=%llu RECORDED=%llu CYCLES=%llu\n", pass, sm,
               work->chunks[sm], work->photons[sm], work->recorded[sm], work->cycles[sm]);
    }
    printf("SMS PASS=%u SMS_USED=%u CYCLES=%llu\n", pass, used, slowest);
    free(work);
}

void free_sm_work(SmWork *sm_work) {
    cudaFree(sm_work);
}
