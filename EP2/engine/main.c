#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "star.h"

#define BATCH_SIZE 1000

typedef struct {
	long long N;
	double a; // 0 to 1
	unsigned int seed;
	unsigned int nphi;
	unsigned int nmu;
	unsigned int nx;
	unsigned int ny;
	unsigned int nspots;
	unsigned int n_observers;
} Config;

Config load_config(int argc, char *argv[]){
	// I am not validating this on purpose. I will have this vlaidated in Python, whoever is using c can understand errors these functions below might raise
	Config config = {
		.N = atoll(argv[1]),
		.seed = atoi(argv[2]),
		.a = atof(argv[3]),
		.nspots = atoi(argv[4]),
		.n_observers = atoi(argv[5]),
		.nx = atoi(argv[6]),
		.ny = atoi(argv[7]),
		.nphi = atoi(argv[8]),
		.nmu = atoi(argv[9]),
	};
	srand(config.seed);
	return config;
}

unsigned int* load_observers_imus(Config config){
	unsigned int* observers_imus = malloc(config.n_observers * sizeof(*observers_imus));
	if (observers_imus == NULL) {
		return NULL;
	}

	if (config.n_observers == 1){
		observers_imus[0] = config.nmu / 2;
		return observers_imus;
	}
	for (unsigned int i = 0; i < config.n_observers; i++){
		observers_imus[i] = i * (config.nmu - 1) / (config.n_observers - 1);
	}
	return observers_imus;

}

void run_simulation(Image image, Config config) {
	printf("Starting simulation with config:  N: %lld, a: %.2f, seed: %d, nphi: %d, nmu: %d, nx: %d, ny: %d, nspots: %d, n_observers: %d\n",  config.N, config.a, config.seed, config.nphi, config.nmu, config.nx, config.ny, config.nspots, config.n_observers);
	SurfaceSpot *spots = get_spots(config.nspots);

	unsigned int* observers_imus = load_observers_imus(config);

	for (long long i=0;i<config.N;i++) {
		sample_photon(image, config.a, spots, config.nspots, config.nx, config.ny, config.nmu, config.nphi, observers_imus, config.n_observers);
	}

	free(observers_imus);
	free(spots);
}

void save_image(Image image, Config config, time_t start_timestamp) {
	char filename[256];
	snprintf(
		filename,
		sizeof(filename),
		"star-N-%lld_seed-%u_a-%.2f_spots-%u_obs-%u_nx-%u_ny-%u_nphi-%u_nmu-%u_t-%lld.bin",
		config.N,
		config.seed,
		config.a,
		config.nspots,
		config.n_observers,
		config.nx,
		config.ny,
		config.nphi,
		config.nmu,
		start_timestamp
	);
	save_image_binary(image, filename);
}

int main(int argc, char *argv[]){
	printf("Started!\n");
	time_t start_timestamp = time(NULL);
	Config config = load_config(argc, argv);
	Image image = get_image(config.N, config.nx, config.ny, config.nphi, config.nmu, config.n_observers);
	run_simulation(image, config);
	printf("saving image\n");
	save_image(image, config, start_timestamp);
	printf("saved image\n");
	free(image.data);
	printf("finished\n");
	return 0;
}