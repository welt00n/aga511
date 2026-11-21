#include <stdio.h>
#include <math.h>

#include "scene.cuh"
#include "rng.cuh"

#define MAX_SPOT_ATTEMPTS 100000
#define PENUMBRA_ESCAPE_PROBABILITY 0.6

static int validate_config(const StarConfig *config) {
    if (config->spots > MAX_SPOTS) {
        fprintf(stderr, "At most %d spots.\n", MAX_SPOTS);
        return -1;
    }
    if (config->nmu > MAX_NMU) {
        fprintf(stderr, "nmu can be at most %d.\n", MAX_NMU);
        return -1;
    }
    if (config->umbra_radius > config->penumbra_radius) {
        fprintf(stderr, "The umbra radius must not be larger than the penumbra radius.\n");
        return -1;
    }
    return 0;
}

static Vec3 random_point_on_sphere(PhiloxState *rng) {
    double z = 2.0 * philox_uniform_double(rng) - 1.0;
    double phi = 2.0 * M_PI * philox_uniform_double(rng);
    double radius = sqrt((1.0 - z) * (1.0 + z));
    Vec3 point = {(real) (radius * cos(phi)), (real) (radius * sin(phi)), (real) z};
    return point;
}

static double angle_between(Vec3 a, Vec3 b) {
    double cosine = (double) a.x * b.x + (double) a.y * b.y + (double) a.z * b.z;
    return acos(fmin(1.0, fmax(-1.0, cosine)));
}

static bool overlaps_placed_spots(Vec3 candidate, const Scene *scene, unsigned int placed, double penumbra_radius) {
    for (unsigned int i = 0; i < placed; i++) {
        if (angle_between(candidate, scene->spot_centers[i]) < 2.0 * penumbra_radius) {
            return true;
        }
    }
    return false;
}

static int place_spots(const StarConfig *config, Scene *scene) {
    PhiloxState rng;
    philox_init(config->seed, SPOTS_SUBSEQUENCE, 0, &rng);
    for (unsigned int placed = 0; placed < config->spots; placed++) {
        int attempts = 0;
        Vec3 candidate;
        do {
            if (++attempts > MAX_SPOT_ATTEMPTS) {
                fprintf(stderr, "Could not place %u spots of radius %g without overlaps.\n",
                        config->spots, config->penumbra_radius);
                return -1;
            }
            candidate = random_point_on_sphere(&rng);
        } while (overlaps_placed_spots(candidate, scene, placed, config->penumbra_radius));
        scene->spot_centers[placed] = candidate;
    }
    scene->spot_count = config->spots;
    return 0;
}

void clear_observer_slots(Scene *scene) {
    for (unsigned int imu = 0; imu < MAX_NMU; imu++) {
        scene->slot_of_imu[imu] = NOT_IN_PASS;
    }
}

static real chord_squared(double angle) {
    return (real) (2.0 * (1.0 - cos(angle)));
}

int build_scene(const StarConfig *config, Scene *scene) {
    if (validate_config(config) != 0) {
        return -1;
    }
    scene->linear_branch_probability = (real) (3.0 / (3.0 + 2.0 * config->limb_darkening));
    scene->umbra_chord_squared = chord_squared(config->umbra_radius);
    scene->penumbra_chord_squared = chord_squared(config->penumbra_radius);
    scene->penumbra_escape_probability = (real) PENUMBRA_ESCAPE_PROBABILITY;
    scene->size = config->size;
    scene->nphi = config->nphi;
    scene->nmu = config->nmu;
    clear_observer_slots(scene);
    return place_spots(config, scene);
}
