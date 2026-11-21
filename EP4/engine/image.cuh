#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <stddef.h>

typedef struct {
    unsigned int *counts;
    size_t pixels;
} Image;

int allocate_image(Image *image, unsigned int nx, unsigned int ny, unsigned int nphi, unsigned int observers);
int save_image(const Image *image, const char *path);
void free_image(Image *image);
size_t image_bytes(const Image *image);

static inline void count_photon(Image *image, unsigned long long pixel) {
    image->counts[pixel]++;
}

#endif
