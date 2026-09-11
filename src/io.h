#ifndef ATTENTION_IO_H
#define ATTENTION_IO_H

#include <limits.h>
#include <stdint.h>

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    exit(EXIT_FAILURE);
}

static void *checked_malloc(size_t bytes) {
    void *ptr = malloc(bytes ? bytes : 1);
    if (!ptr) fail("Allocation failed.");
    return ptr;
}

static void read_header(FILE *file, int dims[4]) {
    if (fread(dims, sizeof(int), 4, file) != 4) fail("Truncated header.");
    for (int i = 0; i < 4; ++i)
        if (dims[i] <= 0) fail("Dimensions must be positive.");
    /* The kernels and MPI collectives use signed int counts. */
    if ((uint64_t)dims[0] * dims[2] > INT_MAX ||
        (uint64_t)dims[0] * dims[3] > INT_MAX ||
        (uint64_t)dims[1] * dims[2] > INT_MAX ||
        (uint64_t)dims[1] * dims[3] > INT_MAX ||
        (uint64_t)dims[2] + dims[3] > INT_MAX ||
        dims[0] > INT_MAX - 511 ||
        dims[2] > INT_MAX / 1024 || dims[3] > INT_MAX / 1024)
        fail("Dimensions exceed supported count limits.");
}

static void read_matrix(double **matrix, size_t count, FILE *file) {
    *matrix = checked_malloc(count * sizeof(double));
    if (fread(*matrix, sizeof(double), count, file) != count)
        fail("Truncated matrix data.");
    for (size_t i = 0; i < count; ++i)
        if (!isfinite((*matrix)[i])) fail("Input must contain finite values.");
}

static void read_matrices(const char *path, double **Q, double **K, double **V,
                         int *m, int *n, int *dk, int *dv) {
    FILE *file = fopen(path, "rb");
    if (!file) fail("Cannot open input file.");
    int dims[4];
    read_header(file, dims);
    *m = dims[0]; *n = dims[1]; *dk = dims[2]; *dv = dims[3];
    read_matrix(Q, (size_t)*m * *dk, file);
    read_matrix(K, (size_t)*n * *dk, file);
    read_matrix(V, (size_t)*n * *dv, file);
    fclose(file);
}

static bool verify(const char *path, const double *result,
                   int m, int n, int dk, int dv) {
    FILE *file = fopen(path, "rb");
    if (!file) fail("Cannot open reference file.");
    int dims[4];
    read_header(file, dims);
    if (dims[0] != m || dims[1] != n || dims[2] != dk || dims[3] != dv)
        fail("Reference dimensions do not match.");
    /* Read past inputs in bounded chunks: no platform-dependent large fseek. */
    size_t left = (size_t)m * dk + (size_t)n * dk + (size_t)n * dv;
    double scratch[1024];
    while (left) {
        size_t count = left < 1024 ? left : 1024;
        if (fread(scratch, sizeof(double), count, file) != count)
            fail("Truncated matrix data.");
        left -= count;
    }
    double max_error = 0.0;
    for (size_t i = 0; i < (size_t)m * dv; ++i) {
        double expected;
        if (fread(&expected, sizeof(double), 1, file) != 1)
            fail("Truncated reference output.");
        double error = fabs(result[i] - expected);
        if (!isfinite(result[i]) || !isfinite(expected) || error > 0.02) {
            fprintf(stderr, "Mismatch at output[%zu]: expected %.9g, got %.9g\n",
                    i, expected, result[i]);
            fclose(file);
            return false;
        }
        if (error > max_error) max_error = error;
    }
    fclose(file);
    printf("Maximum absolute error: %.9g\n", max_error);
    return true;
}
#endif
