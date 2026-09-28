#ifndef BASE_H
#define BASE_H

typedef struct {
    double x;
    double y;
    double z;
} Vec3;

typedef struct {
    unsigned int x;
    unsigned int y;
} PixelCoordinates;

double cartesian2polar(double x, double y);
double generate_random_number();
void set_seed(int argc, char *argv[]);
struct tm* get_now();

#endif