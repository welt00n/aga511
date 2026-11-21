#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "image.cuh"
#include "files.cuh"

static size_t observer_bytes(const Image *image) {
    return image->pixels_per_observer * sizeof(PixelCount);
}

static void touch_every_page(Image *image) {
    volatile PixelCount *counts = image->counts;
    size_t counts_per_page = (size_t) sysconf(_SC_PAGESIZE) / sizeof(PixelCount);
    size_t pixels = (size_t) image->observers * image->pixels_per_observer;
    for (size_t pixel = 0; pixel < pixels; pixel += counts_per_page) {
        counts[pixel] = 0;
    }
}

int allocate_image(Image *image, size_t pixels_per_observer, unsigned int observers) {
    image->pixels_per_observer = pixels_per_observer;
    image->observers = observers;
    image->counts = (PixelCount *) malloc(image_bytes(image));
    if (image->counts == NULL) {
        fprintf(stderr, "Could not allocate the image (%.2f GB).\n", image_bytes(image) / 1e9);
        return -1;
    }
    touch_every_page(image);
    return 0;
}

void zero_image(Image *image, unsigned int observers) {
    memset(image->counts, 0, (size_t) observers * observer_bytes(image));
}

PixelCount *observer_counts(const Image *image, unsigned int slot) {
    return image->counts + (size_t) slot * image->pixels_per_observer;
}

void free_image(Image *image) {
    free(image->counts);
    image->counts = NULL;
}

size_t image_bytes(const Image *image) {
    return (size_t) image->observers * observer_bytes(image);
}

int save_counts(const PixelCount *counts, size_t pixels, const char *path) {
    char partial_path[PARTIAL_PATH_LENGTH];
    partial_path_of(path, partial_path);
    FILE *file = fopen(partial_path, "wb");
    if (file == NULL) {
        perror("Could not open the image file");
        return -1;
    }
    size_t written = fwrite(counts, sizeof(PixelCount), pixels, file);
    int closed = close_on_disk(file);
    if (written != pixels || closed != 0) {
        fprintf(stderr, "Could not write the whole image to %s.\n", partial_path);
        return -1;
    }
    return move_into_place(partial_path, path);
}

unsigned long long photons_in(const PixelCount *counts, size_t pixels) {
    unsigned long long photons = 0;
    for (size_t pixel = 0; pixel < pixels; pixel++) {
        photons += counts[pixel];
    }
    return photons;
}
