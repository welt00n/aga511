#ifndef REPORT_CUH
#define REPORT_CUH

#include "run.cuh"

void report_progress_if_due(const Run *run, Progress *progress);
void print_pass(const Run *run, const Pass *pass, const PassResult *result);
void print_summary(const Run *run, const PhotonCounts *counts, const RunTimes *times, size_t image_bytes);
int write_observer_record(const Run *run, const Pass *pass, const Observer *observer,
                          const PassResult *pass_result, const ObserverResult *result);

#endif
