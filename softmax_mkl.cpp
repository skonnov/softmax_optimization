#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <mkl.h>
#include <mkl_lapack.h>

void softmax(double* vec, int len) {
    double sum = 0.;

    vdExp(len, vec, vec);
    #pragma omp simd reduction (+: sum)
    for (int j = 0; j < len; ++j) {
        sum += vec[j];
    }

    double invert_sum = 1 / sum;
    const long long int cll_len = len;
    const long long int begin_index = 0;
    const long long int step = 1;
    drscl(&cll_len, &sum, vec, &step);
}

int main() {
    int size = 500000000;
    double *check = new double[size];

    int seed = 1010;
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> rand;
    for (int i = 0; i < size; ++i) {
        check[i] = rand(mt);
    }
    double t1 = omp_get_wtime();
    softmax(check, size);
    double t2 = omp_get_wtime();
    std::cout << t2 - t1 << "\n";
    delete[] check;

    return 0;
}