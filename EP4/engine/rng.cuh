#ifndef RNG_CUH
#define RNG_CUH

#include "real.cuh"

#define PHILOX_W32_0   (0x9E3779B9u)
#define PHILOX_W32_1   (0xBB67AE85u)
#define PHILOX_M4x32_0 (0xD2511F53u)
#define PHILOX_M4x32_1 (0xCD9E8D57u)

#define TWO_POW_MINUS_32_FLOAT  (2.3283064e-10f)
#define TWO_POW_MINUS_32_DOUBLE (2.3283064365386963e-10)

typedef struct { unsigned int x, y, z, w; } PhiloxCounter;
typedef struct { unsigned int x, y; } PhiloxKey;

typedef struct {
    PhiloxCounter counter;
    PhiloxCounter block;
    PhiloxKey key;
    unsigned int next_in_block;
} PhiloxState;

static inline HD unsigned int multiply_high_low(unsigned int a, unsigned int b, unsigned int *high) {
    unsigned long long product = (unsigned long long) a * (unsigned long long) b;
    *high = (unsigned int) (product >> 32);
    return (unsigned int) product;
}

static inline HD PhiloxCounter philox_round(PhiloxCounter counter, PhiloxKey key) {
    unsigned int high_0, high_1;
    unsigned int low_0 = multiply_high_low(PHILOX_M4x32_0, counter.x, &high_0);
    unsigned int low_1 = multiply_high_low(PHILOX_M4x32_1, counter.z, &high_1);
    PhiloxCounter mixed = {high_1 ^ counter.y ^ key.x, low_1, high_0 ^ counter.w ^ key.y, low_0};
    return mixed;
}

static inline HD PhiloxCounter philox4x32_10(PhiloxCounter counter, PhiloxKey key) {
    for (int round = 0; round < 9; round++) {
        counter = philox_round(counter, key);
        key.x += PHILOX_W32_0;
        key.y += PHILOX_W32_1;
    }
    return philox_round(counter, key);
}

static inline HD void philox_skip_blocks(PhiloxState *state, unsigned long long blocks) {
    unsigned int low = (unsigned int) blocks;
    unsigned int high = (unsigned int) (blocks >> 32);
    state->counter.x += low;
    if (state->counter.x < low) {
        high++;
    }
    state->counter.y += high;
    if (high <= state->counter.y) {
        return;
    }
    if (++state->counter.z) {
        return;
    }
    ++state->counter.w;
}

static inline HD void philox_next_block(PhiloxState *state) {
    if (++state->counter.x) return;
    if (++state->counter.y) return;
    if (++state->counter.z) return;
    ++state->counter.w;
}

static inline HD void philox_skip_subsequences(PhiloxState *state, unsigned long long subsequences) {
    unsigned int low = (unsigned int) subsequences;
    unsigned int high = (unsigned int) (subsequences >> 32);
    state->counter.z += low;
    if (state->counter.z < low) {
        high++;
    }
    state->counter.w += high;
}

static inline HD void philox_skip_numbers(PhiloxState *state, unsigned long long numbers) {
    state->next_in_block += (unsigned int) (numbers & 3);
    unsigned long long blocks = numbers / 4;
    if (state->next_in_block > 3) {
        blocks += 1;
        state->next_in_block -= 4;
    }
    philox_skip_blocks(state, blocks);
    state->block = philox4x32_10(state->counter, state->key);
}

static inline HD void philox_init(unsigned long long seed, unsigned long long subsequence,
                                  unsigned long long offset, PhiloxState *state) {
    PhiloxCounter zero = {0, 0, 0, 0};
    state->counter = zero;
    state->key.x = (unsigned int) seed;
    state->key.y = (unsigned int) (seed >> 32);
    state->next_in_block = 0;
    philox_skip_subsequences(state, subsequence);
    state->block = philox4x32_10(state->counter, state->key);
    philox_skip_numbers(state, offset);
}

static inline HD unsigned int philox_next(PhiloxState *state) {
    unsigned int number;
    switch (state->next_in_block++) {
    case 0:  number = state->block.x; break;
    case 1:  number = state->block.y; break;
    case 2:  number = state->block.z; break;
    default: number = state->block.w; break;
    }
    if (state->next_in_block == 4) {
        philox_next_block(state);
        state->block = philox4x32_10(state->counter, state->key);
        state->next_in_block = 0;
    }
    return number;
}

static inline HD float philox_uniform_float(PhiloxState *state) {
    return philox_next(state) * TWO_POW_MINUS_32_FLOAT + (TWO_POW_MINUS_32_FLOAT / 2.0f);
}

static inline HD double philox_uniform_double(PhiloxState *state) {
    return philox_next(state) * TWO_POW_MINUS_32_DOUBLE + TWO_POW_MINUS_32_DOUBLE;
}

static inline HD real uniform(PhiloxState *state) {
#ifdef STAR_FP32
    return philox_uniform_float(state);
#else
    return philox_uniform_double(state);
#endif
}

#endif
