#include <stdio.h>
#include <math.h>

#include "observers.cuh"
#include "files.cuh"

#define NO_OBSERVER (-1)

static unsigned int mu_bin_of(double inclination, unsigned int nmu) {
    double fraction = (cos(inclination) + 1.0) / 2.0;
    unsigned int bin = (unsigned int) (fraction * nmu);
    return bin < nmu ? bin : nmu - 1;
}

static double inclination_of_bin_center(unsigned int bin, unsigned int nmu) {
    return acos(-1.0 + (bin + 0.5) * 2.0 / nmu);
}

static Observer describe_observer(unsigned int index, double inclination, unsigned int nmu) {
    Observer observer;
    observer.index = index;
    observer.requested_inclination = inclination;
    observer.mu_bin = mu_bin_of(inclination, nmu);
    observer.bin_center_inclination = inclination_of_bin_center(observer.mu_bin, nmu);
    snprintf(observer.file_stem, FILE_STEM_LENGTH, "observer_mu%04u_i%.4f",
             observer.mu_bin, observer.bin_center_inclination);
    return observer;
}

int locate_observers(const StarConfig *config, ObserverList *list) {
    int observer_in_bin[MAX_NMU];
    for (unsigned int bin = 0; bin < config->nmu; bin++) {
        observer_in_bin[bin] = NO_OBSERVER;
    }
    for (unsigned int index = 0; index < config->observer_count; index++) {
        Observer observer = describe_observer(index, config->inclinations[index], config->nmu);
        int earlier = observer_in_bin[observer.mu_bin];
        if (earlier != NO_OBSERVER) {
            fprintf(stderr, "Observers at %g and %g rad fall in the same mu bin (%u): "
                    "drop one or use a larger nmu.\n",
                    config->inclinations[earlier], observer.requested_inclination, observer.mu_bin);
            return -1;
        }
        observer_in_bin[observer.mu_bin] = (int) index;
        list->observers[index] = observer;
    }
    list->count = config->observer_count;
    return 0;
}

Pass make_pass(unsigned int index, unsigned int observers_per_pass, unsigned int observer_count) {
    Pass pass;
    pass.number = index + 1;
    pass.first_observer = index * observers_per_pass;
    unsigned int remaining = observer_count - pass.first_observer;
    pass.observer_count = remaining < observers_per_pass ? remaining : observers_per_pass;
    return pass;
}

void assign_pass_observers(Scene *scene, const ObserverList *list, const Pass *pass) {
    clear_observer_slots(scene);
    for (unsigned int slot = 0; slot < pass->observer_count; slot++) {
        const Observer *observer = &list->observers[pass->first_observer + slot];
        scene->slot_of_imu[observer->mu_bin] = (int) slot;
    }
}

void observer_path(const char *folder, const Observer *observer, const char *extension, char *path) {
    snprintf(path, PATH_LENGTH, "%s/%s%s", folder, observer->file_stem, extension);
}
