#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "io.h"

// link libraries other than C's math library for you.

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */
void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv) {

    double scale = 1.0 / sqrt((double)dk);

    // Temporary buffer for attention scores per row (length n)
    double* scores = (double*) checked_malloc(sizeof(double) * n);

    for (int i = 0; i < m; i++) {

        /* -----------------------------
         * 1. Compute scores[i][j] = dot(Q[i], K[j]) * scale
         * ----------------------------- */
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            int qi_base = i * dk;
            int kj_base = j * dk;

            for (int t = 0; t < dk; t++) {
                dot += Q[qi_base + t] * K[kj_base + t];
            }
            scores[j] = dot * scale;
        }

        /* -----------------------------
         * 2. Softmax(scores)
         * ----------------------------- */
        double max_val = scores[0];
        for (int j = 1; j < n; j++) {
            if (scores[j] > max_val) max_val = scores[j];
        }

        double sum_exp = 0.0;
        for (int j = 0; j < n; j++) {
            scores[j] = exp(scores[j] - max_val);
            sum_exp += scores[j];
        }
        for (int j = 0; j < n; j++) {
            scores[j] /= sum_exp;
        }

        /* -----------------------------
         * 3. result[i] = scores * V
         * result[i][d] = Σ_j scores[j] * V[j][d]
         * ----------------------------- */
        for (int d = 0; d < dv; d++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += scores[j] * V[j * dv + d];
            }
            result[i * dv + d] = sum;
        }
    }

    free(scores);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testing data>\n", argv[0]);
        return 1;
    }

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
    int m = 0, n = 0, dk = 0, dv = 0;

    read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
    result = checked_malloc(sizeof(double) * (size_t)m * dv);

    struct timespec beg, end;
    clock_gettime(CLOCK_MONOTONIC, &beg);
    attention(Q, K, V, result, m, n, dk, dv);
    clock_gettime(CLOCK_MONOTONIC, &end);

    int ok = verify(argv[1], result, m, n, dk, dv);
    if (ok) {
        double elapsed_time = (end.tv_sec - beg.tv_sec) * 1e6 + (end.tv_nsec - beg.tv_nsec) / 1e3;
        printf("Correct!\nElapsed time: %.2lf us\n", elapsed_time);
    } else {
        puts("Wrong!");
    }

    free(Q);
    free(K);
    free(V);
    free(result);
    return ok ? 0 : 1;
}
