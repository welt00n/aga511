#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <stddef.h>

typedef unsigned int PixelCount;

typedef struct {
    PixelCount *counts;
    size_t pixels_per_observer;
    unsigned int observers;
} Image;

int allocate_image(Image *image, size_t pixels_per_observer, unsigned int observers);
void zero_image(Image *image, unsigned int observers);
PixelCount *observer_counts(const Image *image, unsigned int slot);
void free_image(Image *image);
size_t image_bytes(const Image *image);

int save_counts(const PixelCount *counts, size_t pixels, const char *path);
unsigned long long photons_in(const PixelCount *counts, size_t pixels);

static inline void count_photon(Image *image, unsigned long long pixel) {
    image->counts[pixel]++;
}

#endif
