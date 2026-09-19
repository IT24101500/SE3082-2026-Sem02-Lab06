#include <stdio.h>
#include <omp.h>

/* Original serial version (for reference / correctness check):
 *
 * int fib(int n) {
 *   int i, j;
 *   if (n < 2) return n;
 *   else {
 *     i = fib(n-1);
 *     j = fib(n-2);
 *     return i + j;
 *   }
 * }
 */

int fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n-1) + fib_serial(n-2);
}

int fib_task(int n) {
    int i, j;
    if (n < 2) return n;

    #pragma omp task shared(i) firstprivate(n)
    i = fib_task(n-1);

    #pragma omp task shared(j) firstprivate(n)
    j = fib_task(n-2);

    #pragma omp taskwait
    return i + j;
}

/* Same task parallelization, but stops spawning tasks below a cutoff
 * depth and just calls the fast serial version instead - this avoids
 * creating millions of tiny, overhead-dominated tasks. */
#define CUTOFF 15

int fib_task_cutoff(int n) {
    int i, j;
    if (n < 2) return n;
    if (n < CUTOFF) return fib_serial(n);

    #pragma omp task shared(i) firstprivate(n)
    i = fib_task_cutoff(n-1);

    #pragma omp task shared(j) firstprivate(n)
    j = fib_task_cutoff(n-2);

    #pragma omp taskwait
    return i + j;
}

int main() {
    int n = 30;
    int result_serial, result_parallel, result_cutoff;

    double t0 = omp_get_wtime();
    result_serial = fib_serial(n);
    double t1 = omp_get_wtime();

    double t2 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        result_parallel = fib_task(n);
    }
    double t3 = omp_get_wtime();

    double t4 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        result_cutoff = fib_task_cutoff(n);
    }
    double t5 = omp_get_wtime();

    printf("fib(%d) serial            = %d, time = %f s\n", n, result_serial, t1-t0);
    printf("fib(%d) parallel (no cut) = %d, time = %f s (threads = %d)\n",
           n, result_parallel, t3-t2, omp_get_max_threads());
    printf("fib(%d) parallel (cutoff) = %d, time = %f s (threads = %d)\n",
           n, result_cutoff, t5-t4, omp_get_max_threads());
    printf("Results match: %s\n",
           (result_serial == result_parallel && result_serial == result_cutoff) ? "YES" : "NO");

    return 0;
}
