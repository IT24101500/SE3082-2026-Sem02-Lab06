#include <stdio.h>
#include <omp.h>

#define NPOINTS 1000
#define MAXITER  2000

struct d_complex {
    double r;
    double i;
};

/* Returns 1 if the point escapes (is OUTSIDE the Mandelbrot set)
 * within MAXITER iterations of z = z^2 + c, 0 if it stays bounded
 * (assumed to be INSIDE the set). */
int testpoint(struct d_complex c) {
    struct d_complex z = {0.0, 0.0};
    for (int iter = 0; iter < MAXITER; iter++) {
        double temp = z.r * z.r - z.i * z.i + c.r;
        z.i = 2.0 * z.r * z.i + c.i;
        z.r = temp;
        if (z.r * z.r + z.i * z.i > 4.0) return 1;
    }
    return 0;
}

/* Runs the whole grid scan once with the given number of threads and
 * returns the computed area estimate. Also reports the time taken. */
double run(int nthreads, double *out_time) {
    omp_set_num_threads(nthreads);

    const double eps = 1.0e-5;
    const double a = -2.0, b = 0.5;   /* real axis range   */
    const double c = 0.0,  d = 1.125; /* imaginary axis range (upper half only) */
    long numoutside = 0;

    double t0 = omp_get_wtime();

    /* Start a parallel region before the main loop. i, j and cval are
     * private to each thread (declared inside the loop body), and
     * numoutside is a reduction variable (each thread accumulates its
     * own partial count, which OpenMP then sums together safely). The
     * outermost loop (i) is distributed evenly across the threads. */
    #pragma omp parallel for schedule(dynamic) reduction(+:numoutside)
    for (int i = 0; i < NPOINTS; i++) {
        for (int j = 0; j < NPOINTS; j++) {
            struct d_complex cval;
            cval.r = a + i * (b - a) / NPOINTS + eps;
            cval.i = c + j * (d - c) / NPOINTS + eps;
            numoutside += testpoint(cval);
        }
    }

    double t1 = omp_get_wtime();
    *out_time = t1 - t0;

    long total = (long)NPOINTS * (long)NPOINTS;
    /* Multiply by 2 because we only scanned the upper half of the
     * (symmetric) set. */
    double area = 2.0 * (b - a) * (d - c) * (double)(total - numoutside) / (double)total;
    return area;
}

int main() {
    for (int nt = 1; nt <= 4; nt++) {
        double t;
        double area = run(nt, &t);
        printf("threads = %d, area = %.6f, time = %f s\n", nt, area, t);
    }
    return 0;
}
