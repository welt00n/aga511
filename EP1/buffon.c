#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h> // not even booleans in C, thats cool

// LINK_TO_COLLAB="https://colab.research.google.com/drive/1k3lFfxoS_p-HhlmDtAQS_6qJdhhLcMSy#scrollTo=Vxug-cFT6ym5"
const double PI = 3.14159265358979;
const double L = 1.0;
const double D = 1.0;

double generate_random_number(void) {
    double random_number = rand();
    double max = RAND_MAX;
    return random_number/max;
}

unsigned int now() {
    return (unsigned int) time(NULL);
}

void set_seed(int argc, char *argv[]) {
    unsigned int seed = now();
    if (argc == 3) {
        seed = (unsigned int) atoi(argv[2]);
    }
    srand(seed);
}

bool check_args(int argc, char *argv[]){
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Parametros: N e seed.\n");
        return false;
    }
    if (atoll(argv[1]) <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return false;
    }
    set_seed(argc, argv);
    return true;
}

void run_buffons_experiment(long long N) {
    unsigned int started_at = now();
    long long count = 0;
    for (long long i=0;i<N;i++) {
        double x = generate_random_number() * (D/2.0);
        double theta = generate_random_number() * (PI/2.0);
        if(x<=(L/2.0)*sin(theta)){
            count++;
        }
    }
    double estimate = (2.0*L*N)/(D*count);
    double error = fabs(estimate - PI);
    unsigned int finished_at = now();
    unsigned int elapsed_time = finished_at - started_at;
    printf("PI=%.15f ERROR=%.15e TIME=%d \n", estimate, error, elapsed_time);
}

int main(int argc, char *argv[]){
    if (!check_args(argc, argv)){
        return 1;
    } 
    run_buffons_experiment(atoll(argv[1]));
    return 0;
}
