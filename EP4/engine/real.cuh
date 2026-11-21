#ifndef REAL_CUH
#define REAL_CUH

#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#ifdef STAR_FP32
typedef float real;
#define PRECISION_BITS 32
#else
typedef double real;
#define PRECISION_BITS 64
#endif

#define TWO_PI ((real) 6.28318530717958647692)

static inline HD float  real_sqrt(float x)  { return sqrtf(x); }
static inline HD double real_sqrt(double x) { return sqrt(x); }
static inline HD float  real_cbrt(float x)  { return cbrtf(x); }
static inline HD double real_cbrt(double x) { return cbrt(x); }
static inline HD float  real_sin(float x)   { return sinf(x); }
static inline HD double real_sin(double x)  { return sin(x); }
static inline HD float  real_cos(float x)   { return cosf(x); }
static inline HD double real_cos(double x)  { return cos(x); }
static inline HD float  real_atan2(float y, float x)   { return atan2f(y, x); }
static inline HD double real_atan2(double y, double x) { return atan2(y, x); }

static inline HD real sine_from_cosine(real cosine) {
    real sine_squared = (1 - cosine) * (1 + cosine);
    return sine_squared > 0 ? real_sqrt(sine_squared) : 0;
}

#endif
