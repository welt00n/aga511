#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "base.h"
#include "star.h"

Vec3 sample_surface_point() {
	Vec3 surface_point; 
	surface_point.z = ((2.0 * generate_random_number())-1.0); // Assuming R = 1
	
	double phi = 2.0 * M_PI * generate_random_number();
	double z_squared = surface_point.z * surface_point.z;
	
	double circle_radius = sqrt(1.0 - z_squared);
	surface_point.x = circle_radius*cos(phi);
	surface_point.y = circle_radius*sin(phi);
	return surface_point;
}

Vec3 computeGlobalCoordinatesVector(Vec3 surface_point, double mu_local, double phi_local) {
	double phi = cartesian2polar(surface_point.x, surface_point.y);
	double ze = surface_point.z;

	double sinteta = sqrt(1.0 - mu_local * mu_local);
	double last_term = sqrt(1.0-ze*ze);

	double sin_phi_surface_point = sin(phi);
	double cos_phi_surface_point = cos(phi);

	double sin_phi_local = sin(phi_local);
	double cos_phi_local = cos(phi_local);

	Vec3 global_direction = {
		.x = - sinteta * sin_phi_surface_point * sin_phi_local + cos_phi_surface_point * (sinteta * ze * cos_phi_local + mu_local*last_term),
		.y = sinteta * cos_phi_surface_point * sin_phi_local + sin_phi_surface_point * (sinteta * ze * cos_phi_local + mu_local*last_term),
		.z = mu_local * ze - last_term * sinteta * cos_phi_local
	};
	return global_direction;
}

// Vec3 sample_emission_direction(Vec3 surface_point) {
//     // we gotta sample the direction from the luminosity distribution we except
//     // I=I0*(1+a*mu)
//     // I had to normalize and define the cumulative probability function
	
//     double a = 3.0;
//     double local_xi = generate_random_number();
//     double mu = 
//     double local_mu = (-1 + sqrt(1 + local_xi*a*a + 2.0*local_xi*a)) / a;    
//     double local_phi = 2.0 * M_PI * generate_random_number();

//     Vec3 direction = computeGlobalCoordinatesVector(surface_point, local_mu, local_phi);
//     return direction;
// }

Vec3 sample_emission_direction(double limb_darkening_coef, Vec3 surface_point) {

	const double a = limb_darkening_coef;

	const double sqrt_weight = 3.0 / (3.0 + 2.0 * a);
	
	const double xi_mu = generate_random_number();
	const double xi_component = generate_random_number();

	double local_mu;

	if (xi_component < sqrt_weight) {
		local_mu = sqrt(xi_mu);
	}
	else {
		local_mu = cbrt(xi_mu);
	}

	const double xi_phi = generate_random_number();
	const double local_phi = 2.0 * M_PI * xi_phi;

	Vec3 direction =  computeGlobalCoordinatesVector(surface_point, local_mu, local_phi);

	return direction;
}

int get_photon_intensity(Vec3 surface_point, SurfaceSpot spots[], int spots_count) {
	for (int i = 0; i < spots_count; i ++){
		SurfaceSpot spot = spots[i];
		
		double cos_angle = vec3_dot_product(surface_point, spot.surface_point);
		bool is_inner_point = cos_angle > cos(spot.inner_radius);
		bool is_outter_point = cos_angle > cos(spot.outter_radius);
		
		if (is_inner_point){
			return 0;
		}
		else if (is_outter_point && generate_random_number() > 0.6){
			return 0;
		};
	};
	return 1;
}

Vec3 compute_impact_parameter(Photon photon, double nmu, double nphi) {
	double x = photon.surface_point.x;
	double y = photon.surface_point.y;
	double z = photon.surface_point.z;

	double u = photon.direction.x;
	double v = photon.direction.y;
	double w = photon.direction.z;

	double raiz = sqrt(1.0 - w*w);
	
	double zpp;
	double ypp;
	if (raiz == 0){
		zpp = -x;
		ypp = y;
	}
	else {
		double cf = u / raiz;
		double sf = v / raiz;
		ypp = -sf * x + cf * y;
		zpp = -(cf * x + sf * y) * w + z * raiz;
	}
	Vec3 paramater = {
		.y = ypp,
		.z = zpp
	};
	return paramater;
}

size_t get_image_pixel_index(Image image, unsigned int iphi, unsigned int observer_index, unsigned int ix, unsigned int iy) {
	size_t index = (((size_t)iphi * image.n_observers + observer_index) * image.ny + iy) * image.nx + ix;
	return index;
}

void incrementImagePixel(Image image, Observer observer, unsigned int observer_index, PixelCoordinates pixel_coordinates){
	size_t index = get_image_pixel_index(image, observer.iphi, observer_index, pixel_coordinates.x, pixel_coordinates.y);
	image.data[index]++; // *(image.data + index) += 1; this is also contigous data after all...
}

Observer get_observer(Vec3 photon_direction, unsigned int nmu, unsigned int nphi){
	double tmp = atan2(photon_direction.y, photon_direction.x);
	if (tmp < 0.0) {
		tmp+= 2 * M_PI;
	};
	unsigned int iphi = (unsigned int)((tmp * nphi)/(2.0 * M_PI));
	unsigned int imu = (unsigned int)(0.5 * (photon_direction.z + 1.0) * nmu);
	if (iphi >= nphi) {
		iphi = nphi -1;
	}
	if (imu >= nmu) {
		imu = nmu -1;
	}
	Observer observer = {
		.iphi = iphi,
		.imu = imu
	};
	return observer;
}

PixelCoordinates compute_pixel_coordinates(Photon photon, unsigned int nx, unsigned int ny,  double nmu, double nphi){
	Vec3 impact_parameter = compute_impact_parameter(photon, nmu, nphi);
	double y_rate = ( impact_parameter.y + 1.0 ) / 2.0;
	double z_rate = ( impact_parameter.z + 1.0 ) / 2.0;
	unsigned int ix = (unsigned int)(y_rate * nx);
	unsigned int iy = (unsigned int)(z_rate * ny);
	if (ix >= nx) {
		ix = nx - 1;
	}
	 if (iy >= ny) {
		iy = ny - 1;
	}
	PixelCoordinates pixel_coordinates = {
		.x = ix,
		.y = iy
	};
	return pixel_coordinates;
}

Photon generate_photon(double limb_darkening_coef, SurfaceSpot spots[], int spots_count){
	Photon photon = {
		.surface_point = sample_surface_point(),
	};
	photon.direction = sample_emission_direction(limb_darkening_coef, photon.surface_point),
	photon.intensity = get_photon_intensity(photon.surface_point, spots, spots_count);
	return photon;
}

int find_observer_index(int imu, int observers_imus[], unsigned int n_observers){
	for (unsigned int i = 0; i < n_observers; i++){
		if (imu == observers_imus[i]){
			return i;
		}
	}
	return -1;
}

void sample_photon(Image image, double a, SurfaceSpot spots[], int spots_count, unsigned int nx, unsigned int ny, unsigned int nmu, unsigned int nphi, unsigned int observers_imu[], unsigned int n_observers){
	Photon photon = generate_photon(a, spots, spots_count);
	if (!photon.intensity) {
		return;
	}
	PixelCoordinates pixel_coordinates = compute_pixel_coordinates(photon, nx, ny, nmu, nphi);
	Observer observer = get_observer(photon.direction, nmu, nphi);
	int observer_index = find_observer_index(observer.imu, observers_imu, n_observers);
	if (observer_index == -1){
		return;
	}
	incrementImagePixel(image, observer, observer_index, pixel_coordinates);
}

SurfaceSpot generate_random_spot() {
	SurfaceSpot new_spot;
	new_spot.surface_point = sample_surface_point();
	new_spot.inner_radius = 0.05;
	new_spot.outter_radius = 0.10;
	return new_spot;
}

bool spots_intersect(SurfaceSpot a, SurfaceSpot b) {
	double dot_product = vec3_dot_product(a.surface_point, b.surface_point);
	double angular_distance = acos(dot_product);
	return angular_distance < a.outter_radius + b.outter_radius;
}

void generate_spots(SurfaceSpot* spots, unsigned int count) {
	for (unsigned int i=0; i<count; i++) {
		bool valid = false;
        while (!valid) {
			spots[i] = generate_random_spot();
            valid = true;
            for (unsigned int j=0; j<i; j++) {
				if (spots_intersect(spots[i], spots[j])){
					valid = false;
                    break;
                }
            }
        }
    }
}

SurfaceSpot* get_spots(unsigned int count){
	SurfaceSpot *spots = malloc(count * sizeof(*spots));
	if (spots == NULL) {
		return NULL;
	}
	generate_spots(spots, count);
	return spots;
}
