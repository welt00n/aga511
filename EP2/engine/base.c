#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "base.h"


double cartesian2polar(double x, double y) {
    return atan2(y, x);
}

double get_surface_point_radius(Vec3 surface_point) {
    // nice to check for unit vectors
    double x_squared = surface_point.x * surface_point.x;
    double y_squared = surface_point.y * surface_point.y;
    double z_squared = surface_point.z * surface_point.z;
    return sqrt(x_squared + y_squared + z_squared);
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

void set_seed(int argc, char *argv[]) {
    unsigned int seed = 42;
    if (argc == 3) {
        seed = (unsigned int) atoi(argv[2]);
    }
    srand(seed);
}
