#ifndef PHOTON_CUH
#define PHOTON_CUH

#include "real.cuh"
#include "geometry.cuh"
#include "scene.cuh"
#include "rng.cuh"

#define PHOTON_BLOCKED_BY_SPOT (-1LL)
#define PHOTON_MISSES_OBSERVERS (-2LL)

typedef struct {
    Vec3 position;
    real ring_radius;
    real cos_phi;
    real sin_phi;
} SurfacePoint;

typedef struct {
    unsigned int imu, iphi;
} ObserverBin;

template <typename Rng>
static inline HD SurfacePoint sample_surface_point(Rng *rng) {
    real z = 2 * uniform(rng) - 1;
    real phi = TWO_PI * uniform(rng);
    SurfacePoint point;
    point.ring_radius = sine_from_cosine(z);
    point.cos_phi = real_cos(phi);
    point.sin_phi = real_sin(phi);
    point.position.x = point.ring_radius * point.cos_phi;
    point.position.y = point.ring_radius * point.sin_phi;
    point.position.z = z;
    return point;
}

template <typename Rng>
static inline HD real sample_emission_mu(const Scene *scene, Rng *rng) {
    real xi_mu = uniform(rng);
    real xi_branch = uniform(rng);
    return xi_branch < scene->linear_branch_probability ? real_sqrt(xi_mu) : real_cbrt(xi_mu);
}

template <typename Rng>
static inline HD Vec3 sample_emission_direction(const Scene *scene, SurfacePoint point, Rng *rng) {
    real mu = sample_emission_mu(scene, rng);
    real phi_local = TWO_PI * uniform(rng);
    real sin_theta = sine_from_cosine(mu);
    real cos_local = real_cos(phi_local);
    real sin_local = real_sin(phi_local);
    real z = point.position.z;
    real radial = sin_theta * z * cos_local + mu * point.ring_radius;
    Vec3 direction;
    direction.x = -sin_theta * point.sin_phi * sin_local + point.cos_phi * radial;
    direction.y = sin_theta * point.cos_phi * sin_local + point.sin_phi * radial;
    direction.z = mu * z - point.ring_radius * sin_theta * cos_local;
    return direction;
}

template <typename Rng>
static inline HD bool escapes_spots(const Scene *scene, Vec3 position, Rng *rng) {
    for (unsigned int i = 0; i < scene->spot_count; i++) {
        real distance = distance_squared(position, scene->spot_centers[i]);
        if (distance < scene->umbra_chord_squared) {
            return false;
        }
        if (distance < scene->penumbra_chord_squared) {
            return uniform(rng) <= scene->penumbra_escape_probability;
        }
    }
    return true;
}

static inline HD unsigned int bin_index(real fraction, unsigned int bins) {
    if (!(fraction > 0)) {
        return 0;
    }
    unsigned int index = (unsigned int) (fraction * bins);
    return index < bins ? index : bins - 1;
}

static inline HD ObserverBin observer_bin(const Scene *scene, Vec3 direction) {
    real azimuth = real_atan2(direction.y, direction.x);
    if (azimuth < 0) {
        azimuth += TWO_PI;
    }
    ObserverBin bin;
    bin.iphi = bin_index(azimuth / TWO_PI, scene->nphi);
    bin.imu = bin_index((direction.z + 1) / 2, scene->nmu);
    return bin;
}

static inline HD unsigned long long pixel_index(const Scene *scene, Vec3 position, Vec3 direction,
                                                ObserverBin bin, int observer) {
    real u = direction.x, v = direction.y, w = direction.z;
    real root = sine_from_cosine(w);
    real ypp, zpp;
    if (root == 0) {
        zpp = -position.x;
        ypp = position.y;
    } else {
        real cf = u / root;
        real sf = v / root;
        ypp = -sf * position.x + cf * position.y;
        zpp = -(cf * position.x + sf * position.y) * w + position.z * root;
    }
    unsigned int ix = bin_index((ypp + 1) / 2, scene->nx);
    unsigned int iy = bin_index((zpp + 1) / 2, scene->ny);
    unsigned long long frame = (unsigned long long) bin.iphi * scene->observers + (unsigned int) observer;
    return (frame * scene->ny + iy) * scene->nx + ix;
}

template <typename Rng>
static inline HD long long trace_photon(const Scene *scene, Rng *rng) {
    SurfacePoint point = sample_surface_point(rng);
    Vec3 direction = sample_emission_direction(scene, point, rng);
    if (!escapes_spots(scene, point.position, rng)) {
        return PHOTON_BLOCKED_BY_SPOT;
    }
    ObserverBin bin = observer_bin(scene, direction);
    int observer = scene->observer_of_imu[bin.imu];
    if (observer == NOT_AN_OBSERVER) {
        return PHOTON_MISSES_OBSERVERS;
    }
    return (long long) pixel_index(scene, point.position, direction, bin, observer);
}

#endif
