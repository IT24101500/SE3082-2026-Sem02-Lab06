#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000
/* Strip size chosen to match a 256-bit SIMD register width for 32-bit
 * floats (256 / 32 = 8 floats per vector instruction, e.g. AVX2). */
#define STRIP 8

int main() {
    float *A = malloc(N * sizeof(float));
    float *B = malloc(N * sizeof(float));
    float *C = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.5f;
        B[i] = (float)(i % 1000) * 2.0f;
    }

    double t0 = omp_get_wtime();

    /* Strip mining: split the N-element loop into fixed-size strips of
     * STRIP elements. Each strip is a small, cache- and SIMD-friendly
     * unit of work. The outer loop over strips is distributed across
     * OpenMP threads, and the inner loop over one strip is marked for
     * SIMD vectorization by the compiler. */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < N; s += STRIP) {
        int end = (s + STRIP < N) ? s + STRIP : N;
        #pragma omp simd
        for (int i = s; i < end; i++) {
            C[i] = A[i] * B[i];
        }
    }

    double t1 = omp_get_wtime();

    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] * B[i]) { ok = 0; break; }
    }

    printf("N = %d, STRIP = %d, threads = %d\n", N, STRIP, omp_get_max_threads());
    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");
    printf("Time taken: %f s\n", t1 - t0);

    free(A);
    free(B);
    free(C);
    return 0;
}
