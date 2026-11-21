#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.cuh"
#include "files.cuh"

static const PixelCount *observer_block(const Image *image, unsigned int slot) {
    return image->counts + (size_t) slot * image->pixels_per_observer;
}

int allocate_image(Image *image, size_t pixels_per_observer, unsigned int observers) {
    image->pixels_per_observer = pixels_per_observer;
    image->observers = observers;
    image->counts = (PixelCount *) malloc(image_bytes(image));
    if (image->counts == NULL) {
        fprintf(stderr, "Could not allocate the image (%.2f GB).\n", image_bytes(image) / 1e9);
        return -1;
    }
    return 0;
}

void zero_image(Image *image, unsigned int observers) {
    memset(image->counts, 0, (size_t) observers * observer_bytes(image));
}

int save_observer(const Image *image, unsigned int slot, const char *path) {
    char partial_path[PARTIAL_PATH_LENGTH];
    partial_path_of(path, partial_path);
    FILE *file = fopen(partial_path, "wb");
    if (file == NULL) {
        perror("Could not open the image file");
        return -1;
    }
    size_t written = fwrite(observer_block(image, slot), sizeof(PixelCount), image->pixels_per_observer, file);
    int closed = fclose(file);
    if (written != image->pixels_per_observer || closed != 0) {
        fprintf(stderr, "Could not write the whole image to %s.\n", partial_path);
        return -1;
    }
    return move_into_place(partial_path, path);
}

unsigned long long photons_in_observer(const Image *image, unsigned int slot) {
    const PixelCount *block = observer_block(image, slot);
    unsigned long long photons = 0;
    for (size_t pixel = 0; pixel < image->pixels_per_observer; pixel++) {
        photons += block[pixel];
    }
    return photons;
}

void free_image(Image *image) {
    free(image->counts);
    image->counts = NULL;
}

size_t image_bytes(const Image *image) {
    return (size_t) image->observers * observer_bytes(image);
}

size_t observer_bytes(const Image *image) {
    return image->pixels_per_observer * sizeof(PixelCount);
}
