#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.cuh"

int allocate_image(Image *image, unsigned int nx, unsigned int ny, unsigned int nphi, unsigned int observers) {
    image->pixels = (size_t) nx * ny * nphi * observers;
    image->counts = (unsigned int *) malloc(image_bytes(image));
    if (image->counts == NULL) {
        fprintf(stderr, "Could not allocate the image (%.2f GB).\n", image_bytes(image) / 1e9);
        return -1;
    }
    memset(image->counts, 0, image_bytes(image));  // not calloc: zero every page now, inside the timed setup
    return 0;
}

int save_image(const Image *image, const char *path) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror("Could not open the image file");
        return -1;
    }
    size_t written = fwrite(image->counts, sizeof(*image->counts), image->pixels, file);
    int closed = fclose(file);
    if (written != image->pixels || closed != 0) {
        fprintf(stderr, "Could not write the whole image to %s.\n", path);
        return -1;
    }
    return 0;
}

void free_image(Image *image) {
    free(image->counts);
    image->counts = NULL;
}

size_t image_bytes(const Image *image) {
    return image->pixels * sizeof(*image->counts);
}
