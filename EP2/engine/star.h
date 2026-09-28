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

typedef struct {
    // pointer to the ints we are accumulating to
    unsigned int *data;     
    // how many we will simulate
    unsigned int N;
    // image grid  
    unsigned int nx; 
    unsigned int ny;
    // viewing angles split in phi is along the azimutal angle
    unsigned int nmu;
    unsigned int nphi;
} Image;

Image get_image(unsigned int nx, unsigned int ny, unsigned int nphi, unsigned int nmu);
Vec3 sample_surface_point();
Vec3 computeGlobalCoordinatesVector(Vec3 surface_point, double mu_local, double phi_local);
Vec3 sample_emission_direction(Vec3 surface_point);
Photon generate_photon(SurfaceSpot spots[], int spots_count);

Observer get_observer(Vec3 photon_direction, unsigned int nmu, unsigned int nphi);
Vec3 compute_impact_parameter(Photon photon, Observer observer, double nmu, double nphi);
PixelCoordinates computer_pixel_coordinates(Photon photon, Observer observer, unsigned int nx, unsigned int ny,  double nmu, double nphi);

size_t get_image_pixel_index(Image image, unsigned int iphi, unsigned int imu, unsigned int ix, unsigned int iy);
void incrementImagePixel(Image image, Observer observer, PixelCoordinates pixel_coordinates);
double get_surface_point_radius(Vec3 surface_point);

void sample_photon(Image image, SurfaceSpot spots[], int spots_count, unsigned int nx, unsigned int ny, double nmu, double nphi);
void save_image_binary(Image image);