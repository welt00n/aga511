#ifndef FILES_CUH
#define FILES_CUH

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define PATH_LENGTH 4096
#define PARTIAL_SUFFIX ".partial"
#define PARTIAL_PATH_LENGTH (PATH_LENGTH + sizeof(PARTIAL_SUFFIX))

static inline void partial_path_of(const char *path, char *partial_path) {
    snprintf(partial_path, PARTIAL_PATH_LENGTH, "%s" PARTIAL_SUFFIX, path);
}

static inline int close_on_disk(FILE *file) {
    bool on_disk = fflush(file) == 0 && fsync(fileno(file)) == 0;
    bool closed = fclose(file) == 0;
    return on_disk && closed ? 0 : -1;
}

static inline int sync_parent_folder(const char *path) {
    char folder[PATH_LENGTH];
    snprintf(folder, PATH_LENGTH, "%s", path);
    char *last_slash = strrchr(folder, '/');
    if (last_slash == NULL) {
        snprintf(folder, PATH_LENGTH, ".");
    } else {
        *last_slash = '\0';
    }
    int descriptor = open(folder, O_RDONLY);
    if (descriptor < 0) {
        return -1;
    }
    int synced = fsync(descriptor);
    close(descriptor);
    return synced;
}

static inline int move_into_place(const char *partial_path, const char *path) {
    if (rename(partial_path, path) != 0 || sync_parent_folder(path) != 0) {
        perror("Could not move the finished file into place");
        return -1;
    }
    return 0;
}

#endif
