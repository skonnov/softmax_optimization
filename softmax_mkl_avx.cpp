#define AVX2

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <mkl.h>
#include <immintrin.h>


void softmax(double* vec, int len) {
    double sum = 0.;
    vdExp(len, vec, vec);

    __m512d simd_sum = _mm512_set1_pd(0.0f);
    __m512d x;
    int j = 0;
    int align_len = len - len % 8;
    for (j = 0; j < align_len; j+= 8) {
        x = _mm512_loadu_pd(vec + j);
        simd_sum = _mm512_add_pd(simd_sum, x);
    }

    double simd_sum_double_ptr[8];
    _mm512_storeu_pd(simd_sum_double_ptr, simd_sum);

    for (int i = 0; i < 8; ++i) {
        sum += simd_sum_double_ptr[i];
    }

    for (int k = j; k < len; ++k) {
        sum += vec[k];
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
    std::uniform_real_distribution<double> rand(0., 1.);
    for (int i = 0; i < size; ++i) {
        check[i] = rand(mt);
    }
    double t1 = omp_get_wtime();
    softmax(check, size);
    double t2 = omp_get_wtime();
    std::cout << t2 - t1 << "\n";
    delete[] check;
    // for (auto y: res) {
    //     std::cout << y << " ";
    // }
    // std::cout << "\n";
    return 0;
}