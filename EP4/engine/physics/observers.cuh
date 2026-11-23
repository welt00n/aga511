#ifndef OBSERVERS_CUH
#define OBSERVERS_CUH

#include "physics/scene.cuh"

#define FILE_STEM_LENGTH 64

typedef struct {
    unsigned int index;
    double requested_inclination;
    double bin_center_inclination;
    unsigned int mu_bin;
    char file_stem[FILE_STEM_LENGTH];
} Observer;

typedef struct {
    unsigned int count;
    Observer observers[MAX_OBSERVERS];
} ObserverList;

typedef struct {
    unsigned int number;
    unsigned int first_observer;
    unsigned int observer_count;
} Pass;

int locate_observers(const StarConfig *config, ObserverList *list);
Pass make_pass(unsigned int index, unsigned int observers_per_pass, unsigned int observer_count);
void assign_pass_observers(Scene *scene, const ObserverList *list, const Pass *pass);
void observer_path(const char *folder, const Observer *observer, const char *extension, char *path);

#endif
