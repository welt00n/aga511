#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "physics/real.cuh"

typedef struct {
    real x, y, z;
} Vec3;

static inline HD real distance_squared(Vec3 a, Vec3 b) {
    real dx = a.x - b.x;
    real dy = a.y - b.y;
    real dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

#endif
