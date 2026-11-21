#ifndef SCENE_CUH
#define SCENE_CUH

#include "geometry.cuh"

#define MAX_SPOTS 64
#define MAX_NMU 4096
#define NOT_AN_OBSERVER (-1)
#define SPOTS_SUBSEQUENCE (~0ULL)

typedef struct {
    long long photons;
    unsigned long long seed;
    double limb_darkening;
    unsigned int spots;
    double umbra_radius;
    double penumbra_radius;
    unsigned int observers;
    unsigned int nx, ny;
    unsigned int nphi, nmu;
} StarConfig;

typedef struct {
    real linear_branch_probability;
    unsigned int spot_count;
    Vec3 spot_centers[MAX_SPOTS];
    real umbra_chord_squared;
    real penumbra_chord_squared;
    real penumbra_escape_probability;
    unsigned int nx, ny, nphi, nmu, observers;
    int observer_of_imu[MAX_NMU];
} Scene;

int build_scene(const StarConfig *config, Scene *scene);

#endif
