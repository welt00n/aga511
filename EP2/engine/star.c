#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "base.h"
#include "star.h"

Image get_image(unsigned int nx, unsigned int ny, unsigned int nphi, unsigned int nmu) {
	Image image ;
	image.nx = nx;
	image.ny = ny;
	image.nphi = nphi;
	image.nmu = nmu;

	size_t count = (size_t)nx*ny*nphi*nmu;
	image.data = calloc(count, sizeof(*image.data));
	if (image.data == NULL) {
		printf("Allocation failed");
	}
	return image;  
}

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

Vec3 sample_emission_direction(Vec3 surface_point) {

	const double a = 1.0;

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

double vec3_dot_product(Vec3 vec1, Vec3 vec2) {
	return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
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

Photon generate_photon(SurfaceSpot spots[], int spots_count){
	Photon photon = {
		.surface_point = sample_surface_point(),
	};
	photon.direction = sample_emission_direction(photon.surface_point),
	photon.intensity = get_photon_intensity(photon.surface_point, spots, spots_count);
	return photon;
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

Vec3 compute_impact_parameter(Photon photon, Observer observer, double nmu, double nphi) {
	double x = photon.surface_point.x;
	double y = photon.surface_point.y;
	double z = photon.surface_point.z;


	// // // Observer direction
	// double mu_observer =
	// 	-1.0 + 2.0 * ((double)observer.imu + 0.5) / nmu;

	// double phi_observer =
	// 	2.0 * M_PI * ((double)observer.iphi + 0.5) / nphi;

	// double sin_theta =
	// 	sqrt(1.0 - mu_observer * mu_observer);

	// double u = sin_theta * cos(phi_observer);
	// double v = sin_theta * sin(phi_observer);
	// double w = mu_observer;


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

PixelCoordinates computer_pixel_coordinates(Photon photon, Observer observer, unsigned int nx, unsigned int ny,  double nmu, double nphi){
	Vec3 impact_parameter = compute_impact_parameter(photon, observer, nmu, nphi);
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

size_t get_image_pixel_index(Image image, unsigned int iphi, unsigned int imu, unsigned int ix, unsigned int iy) {
	size_t index = (((size_t)iphi * image.nmu + imu) * image.ny + iy) * image.nx + ix;
	return index;
}

void incrementImagePixel(Image image, Observer observer, PixelCoordinates pixel_coordinates){
	size_t index = get_image_pixel_index(image, observer.iphi, observer.imu, pixel_coordinates.x, pixel_coordinates.y);
	image.data[index]++; // *(image.data + index) += 1; this is also contigous data after all...
}

void sample_photon(Image image, SurfaceSpot spots[], int spots_count, unsigned int nx, unsigned int ny, double nmu, double nphi){
	Photon photon = generate_photon(spots, spots_count);
	if (!photon.intensity){
		return;
	}
	Observer observer = get_observer(photon.direction, nmu, nphi);
	PixelCoordinates pixel_coordinates = computer_pixel_coordinates(photon, observer, nx, ny, nmu, nphi);
	incrementImagePixel(image, observer, pixel_coordinates);
}

void save_image_binary(Image image) {    
	struct tm *current_time = get_now();
	
	char filename[128];
	strftime(
		filename,
		sizeof(filename),
		"star_%Y-%m-%d_%H-%M-%S.bin",
		current_time
	);

	FILE *file_stream = fopen(filename, "wb");
	if (file_stream == NULL) {
		perror("Could not open image file"); // there are two ways to print an error in the same fund =-D
		return;
	}

	size_t count = (size_t)image.nx * image.ny * image.nphi * image.nmu;
	size_t written = fwrite(image.data, sizeof(*image.data), count, file_stream);
	fclose(file_stream);

	if (written != count) {
		fprintf(stderr, "Error writing image data\n"); // there are two ways to print an error in the same fund =-D
		return;
	}
	return;
}