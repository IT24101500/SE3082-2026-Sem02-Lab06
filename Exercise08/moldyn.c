/*
 * Molecular Dynamics simulation of argon atoms (Lennard-Jones fluid)
 * with periodic boundary conditions, parallelized with OpenMP.
 *
 * Reduced (Lennard-Jones) units are used throughout: sigma = 1,
 * epsilon = 1, mass = 1, kB = 1.
 *
 * The main loop follows the classic structure:
 *   domove()  - move particles using current velocities/forces,
 *               partially update velocities
 *   forces()  - compute LJ forces at the new positions, accumulate
 *               potential energy and virial (PARALLELIZED)
 *   mkekin()  - finish the velocity update, compute kinetic energy
 *   velavg()  - compute average temperature, apply simple velocity
 *               rescaling thermostat
 *   prnout()  - print diagnostics
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define NC   8                 /* unit cells per dimension        */
#define N    (4 * NC * NC * NC)/* number of atoms (FCC: 4/cell)   */
#define DENSITY 0.75
#define DT      0.001
#define RCUT    2.5
#define TEMP0   1.0
#define NSTEPS  20
#define PRINT_EVERY 5

static double x[N][3], v[N][3], f[N][3];
static double L;               /* box length */

static double minimum_image(double d) {
    d -= L * floor(d / L + 0.5);
    return d;
}

static void init(void) {
    double a = pow(4.0 / DENSITY, 1.0 / 3.0); /* FCC lattice constant */
    L = a * NC;

    /* FCC basis: 4 atoms per unit cell */
    double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    int idx = 0;
    for (int ix = 0; ix < NC; ix++)
        for (int iy = 0; iy < NC; iy++)
            for (int iz = 0; iz < NC; iz++)
                for (int b = 0; b < 4; b++) {
                    x[idx][0] = (ix + basis[b][0]) * a;
                    x[idx][1] = (iy + basis[b][1]) * a;
                    x[idx][2] = (iz + basis[b][2]) * a;
                    idx++;
                }

    srand(12345);
    double vsum[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            v[i][k] = ((double)rand() / RAND_MAX) - 0.5;
            vsum[k] += v[i][k];
        }
    }
    /* remove net momentum so the whole system does not drift */
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            v[i][k] -= vsum[k] / N;

    /* rescale velocities so the instantaneous temperature is TEMP0 */
    double ke = 0.0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            ke += v[i][k] * v[i][k];
    double temp = ke / (3.0 * N);
    double scale = sqrt(TEMP0 / temp);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            v[i][k] *= scale;

    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            f[i][k] = 0.0;
}

static void domove(void) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            v[i][k] += 0.5 * DT * f[i][k];   /* first half of velocity update */
            x[i][k] += DT * v[i][k];         /* move particle                */
            x[i][k] -= L * floor(x[i][k] / L); /* wrap into periodic box     */
        }
    }
}

/*
 * forces(): the outer loop over particle i is parallelized with
 * "#pragma omp parallel for". Each thread owns a whole range of i
 * values, so f[i] is only ever touched by one thread. However each
 * inner iteration also updates f[j] for j > i, and j can belong to
 * ANY thread's range - multiple threads can therefore try to update
 * the SAME f[j] element at the same time. That update is protected
 * with "#pragma omp critical".
 *
 * "pot" (potential energy) and "vir" (virial) are the two REDUCTION
 * variables: each thread accumulates its own partial sum and OpenMP
 * combines them safely at the end.
 */
static void forces(int chunk, double *pot_out, double *vir_out) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            f[i][k] = 0.0;

    double pot = 0.0, vir = 0.0;
    double rc2 = RCUT * RCUT;

    #pragma omp parallel for schedule(static, chunk) reduction(+:pot,vir)
    for (int i = 0; i < N - 1; i++) {
        double fi[3] = {0.0, 0.0, 0.0};
        for (int j = i + 1; j < N; j++) {
            double dx = minimum_image(x[i][0] - x[j][0]);
            double dy = minimum_image(x[i][1] - x[j][1]);
            double dz = minimum_image(x[i][2] - x[j][2]);
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < rc2 && r2 > 1.0e-12) {
                double sr2 = 1.0 / r2;
                double sr6 = sr2 * sr2 * sr2;
                double sr12 = sr6 * sr6;
                double fr = 24.0 * (2.0 * sr12 - sr6) * sr2; /* F/r */

                fi[0] += fr * dx;
                fi[1] += fr * dy;
                fi[2] += fr * dz;

                /* f[j] is shared with other threads: must be atomic/critical */
                #pragma omp critical
                {
                    f[j][0] -= fr * dx;
                    f[j][1] -= fr * dy;
                    f[j][2] -= fr * dz;
                }

                pot += 4.0 * (sr12 - sr6);
                vir += fr * r2;
            }
        }
        /* f[i] is only ever written by the thread that owns index i,
         * so no synchronization is needed here. */
        f[i][0] += fi[0];
        f[i][1] += fi[1];
        f[i][2] += fi[2];
    }

    *pot_out = pot;
    *vir_out = vir;
}

static double mkekin(void) {
    double ke = 0.0;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            v[i][k] += 0.5 * DT * f[i][k]; /* second half of velocity update */
            ke += v[i][k] * v[i][k];
        }
    }
    return 0.5 * ke;
}

static double velavg(void) {
    double ke = 0.0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            ke += v[i][k] * v[i][k];
    return ke / (3.0 * N); /* instantaneous temperature */
}

static void prnout(int step, double pot, double ke, double temp) {
    printf("step %3d  pot = %10.4f  kin = %10.4f  total = %10.4f  temp = %8.4f\n",
           step, pot / N, ke / N, (pot + ke) / N, temp);
}

int main(int argc, char **argv) {
    int chunk = (argc > 1) ? atoi(argv[1]) : 4;

    init();

    printf("N = %d atoms, box length L = %.4f, threads = %d, schedule(static,%d)\n",
           N, L, omp_get_max_threads(), chunk);

    double t0 = omp_get_wtime();
    for (int step = 1; step <= NSTEPS; step++) {
        domove();
        double pot, vir;
        forces(chunk, &pot, &vir);
        double ke = mkekin();
        double temp = velavg();

        if (step % PRINT_EVERY == 0 || step == 1)
            prnout(step, pot, ke, temp);
    }
    double t1 = omp_get_wtime();

    printf("Time taken: %f s\n", t1 - t0);
    return 0;
}
