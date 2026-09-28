#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "star.h"

#define BATCH_SIZE 1000

bool check_args(int argc, char *argv[]){
    // if (argc < 2 || argc > 3) {
    //     fprintf(stderr, "Parametros: N e seed.\n");
    //     return false;
    // }
    if (atoll(argv[1]) <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return false;
    }
    set_seed(argc, argv);
    return true;
}

typedef struct {
    unsigned int N;
    unsigned int seed;
    unsigned int nphi;
    unsigned int nmu;
    unsigned int nx;
    unsigned int ny;
} Config;

Config load_config(int argc, char *argv[]){
    Config config;
    if (!check_args(argc, argv)){ // gotta improve check_args, but thats better than nothing for now.
        return config ;
    }
    config.N = atoll(argv[1]);
    config.seed = atoll(argv[2]);
    config.nphi = atoll(argv[3]); // number of frames??
    config.nmu = atoll(argv[4]);
    config.nx = atoll(argv[5]);
    config.ny = atoll(argv[6]);
    return config;
}

int main(int argc, char *argv[]){
    Config config = load_config(argc, argv);
    
    Image image = get_image(config.nx, config.ny, config.nphi, config.nmu);
    int spots_count = 8;
    SurfaceSpot spots[8] = {
        {
            .surface_point = {0.35, 0.5, 0.8},
            .inner_radius = 0.05,
            .outter_radius = 0.07
        },
        {
            .surface_point = {-0.5, -0.35, -0.8},
            .inner_radius = 0.05,
            .outter_radius = 0.07
        },
        {
            .surface_point = {0.5, -0.35, -0.8},
            .inner_radius = 0.05,
            .outter_radius = 0.07
        },
         {
            .surface_point = {0.5, 0.35, -0.8},
            .inner_radius = 0.05,
            .outter_radius = 0.07
        },
        {
            .surface_point = {1.0, 0.0, 0.0},
            .inner_radius = 0.01,
            .outter_radius = 0.05
        },
        {
            .surface_point = {0.0, 0.0, 1.0},
            .inner_radius = 0.02,
            .outter_radius = 0.04
        },
        {
            .surface_point = {-1.0, 0.0, 0.0},
            .inner_radius = 0.01,
            .outter_radius = 0.05
        },
        {
            .surface_point = {0.0, 0.0, -1.0},
            .inner_radius = 0.02,
            .outter_radius = 0.04
        }
    };

    for (long long i=0;i<config.N;i++) {
        sample_photon(image, spots, spots_count, config.nx, config.ny, config.nmu, config.nphi);
    }
    save_image_binary(image);
    
    free(image.data);
    return 0;
}
