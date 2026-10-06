#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "base.h"

double cartesian2polar(double x, double y) {
	return atan2(y, x);
}

double vec3_dot_product(Vec3 vec1, Vec3 vec2) {
	return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

double generate_random_number(void) {
	double random_number = rand();
	double max = RAND_MAX;
	return random_number/max;
}

struct tm* get_now() {
	time_t now = time(NULL);
	struct tm *current_time = localtime(&now);
	return current_time;
}

Image get_image(unsigned int N, unsigned int nx, unsigned int ny, unsigned int nphi, unsigned int nmu, unsigned int angles) {
	Image image ;
	image.N = N;
	image.nx = nx;
	image.ny = ny;
	image.nphi = nphi;
	image.nmu = nmu;
	image.n_observers = angles;

	size_t count = (size_t)nx*ny*nphi*angles;
	image.data = calloc(count, sizeof(*image.data));
	if (image.data == NULL) {
		printf("Allocation failed");
	}
	return image;
}

void save_image_binary(Image image, char* filename) {
	FILE *file_stream = fopen(filename, "wb");
	if (file_stream == NULL) {
		perror("Could not open image file"); // there are two ways to print an error in the same fund =-D
		return;
	}

	size_t count = (size_t)image.nx * image.ny * image.nphi * image.n_observers;
	size_t written = fwrite(image.data, sizeof(*image.data), count, file_stream);
	fclose(file_stream);

	if (written != count) {
		fprintf(stderr, "Error writing image data\n"); // there are two ways to print an error in the same fund =-D
		return;
	}

	return;
}