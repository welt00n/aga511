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
    unsigned int n_observers;
} Image;

double cartesian2polar(double x, double y);
double generate_random_number();
double vec3_dot_product(Vec3 vec1, Vec3 vec2);

Image get_image(unsigned int N, unsigned int nx, unsigned int ny, unsigned int nphi, unsigned int nmu, unsigned int angles);
void save_image_binary(Image image, char* filename);

#endif