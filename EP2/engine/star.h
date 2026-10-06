#include "base.h"

typedef struct {
    Vec3 surface_point;
    Vec3 direction;
    int intensity;
} Photon;

typedef struct {
    Vec3 surface_point;
    double inner_radius; // rad
    double outter_radius;
} SurfaceSpot;

typedef struct {
    unsigned int imu;
    unsigned int iphi;
} Observer;

void sample_photon(Image image, double a, SurfaceSpot spots[], int spots_count, unsigned int nx, unsigned int ny, unsigned int nmu, unsigned int nphi, unsigned int angles[], unsigned int n_angles);
SurfaceSpot* get_spots(unsigned int count);
