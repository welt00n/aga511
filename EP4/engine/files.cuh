#ifndef FILES_CUH
#define FILES_CUH

#include <stdio.h>

#define PATH_LENGTH 4096
#define PARTIAL_SUFFIX ".partial"
#define PARTIAL_PATH_LENGTH (PATH_LENGTH + sizeof(PARTIAL_SUFFIX))

static inline void partial_path_of(const char *path, char *partial_path) {
    snprintf(partial_path, PARTIAL_PATH_LENGTH, "%s" PARTIAL_SUFFIX, path);
}

static inline int move_into_place(const char *partial_path, const char *path) {
    if (rename(partial_path, path) != 0) {
        perror("Could not rename the finished file");
        return -1;
    }
    return 0;
}

#endif
