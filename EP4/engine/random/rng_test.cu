#include <stdio.h>

#include "random/rng_test.cuh"
#include "random/rng.cuh"
#include "processor.cuh"

#define TEST_SEED 42

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

static bool same_counter(PhiloxCounter a, PhiloxCounter b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

static int test_known_answers(void) {
    int failures = 0;
    int tests = sizeof(RANDOM123_KNOWN_ANSWERS) / sizeof(RANDOM123_KNOWN_ANSWERS[0]);
    for (int i = 0; i < tests; i++) {
        PhiloxKnownAnswer test = RANDOM123_KNOWN_ANSWERS[i];
        PhiloxCounter got = philox4x32_10(test.counter, test.key);
        bool passed = same_counter(got, test.expected);
        printf("PHILOX_TEST_%d=%s GOT=%08x,%08x,%08x,%08x\n", i, passed ? "PASS" : "FAIL", got.x, got.y, got.z, got.w);
        failures += passed ? 0 : 1;
    }
    return failures;
}

static void print_first_numbers_of_chunk_0(unsigned long long seed) {
    PhiloxState rng;
    philox_init(seed, 0, 0, &rng);
    printf("SEED_%llu_CHUNK_0=", seed);
    for (int i = 0; i < 8; i++) {
        printf("%08x%s", philox_next(&rng), i < 7 ? "," : "\n");
    }
}

int test_rng(void) {
    int failures = test_known_answers();
    print_first_numbers_of_chunk_0(TEST_SEED);
    failures += compare_rng_with_cpu(TEST_SEED);
    printf("PHILOX_TESTS=%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
