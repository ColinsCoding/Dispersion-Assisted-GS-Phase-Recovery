/* burgers.cu -- viscous Burgers u_t + u u_x = nu u_xx on [0,1), periodic,
 * u0 = sin(2 pi x), for B independent viscosities at once.
 *
 * Same scheme as dgs/lagrangian_fluids.py (central differences, conservative
 * flux, RK4) so results agree with it to roundoff. Each RK4 stage is ONE fused
 * kernel (right-hand side + stage update + accumulator), so a time step is 4
 * kernel launches. The torch version issues ~40 small kernels per step.
 *
 * Row b uses nu_b = 0.01 + 0.04 b / (B - 1)   (row 0: nu = 0.01, last row: 0.05).
 *
 * Build (x64 Native Tools prompt, or after vcvars64.bat):
 *     nvcc -O2 -o burgers.exe burgers.cu
 * Run:
 *     burgers.exe B N t_end            solve to t_end, write burgers_out.csv
 *     burgers.exe B N t_end n_steps    benchmark only: time n_steps steps
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

__global__ void stage(const double *u, const double *us, double *acc, double *out,
                      const double *nus, int N, int B, double dx,
                      double cdt, double w, int first, int last, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    int b = idx / N, i = idx % N;
    int ip = b * N + (i + 1) % N, im = b * N + (i + N - 1) % N;
    double up = us[ip], um = us[im], uc = us[idx];
    double k = -(up * up - um * um) / (4.0 * dx) + nus[b] * (up - 2.0 * uc + um) / (dx * dx);
    double a = (first ? 0.0 : acc[idx]) + w * k;
    if (!last) { acc[idx] = a; out[idx] = u[idx] + cdt * k; }
    else       { out[idx] = u[idx] + dt / 6.0 * a; }
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: burgers B N t_end [n_steps]\n"); return 1; }
    int B = atoi(argv[1]), N = atoi(argv[2]);
    double t_end = atof(argv[3]);
    int bench = argc >= 5 ? atoi(argv[4]) : 0;
    if (B < 1 || N < 16 || t_end <= 0) { fprintf(stderr, "need B >= 1, N >= 16, t_end > 0\n"); return 1; }

    std::vector<double> nus(B), u0((size_t)B * N);
    double numax = 0;
    for (int b = 0; b < B; b++) {
        nus[b] = B > 1 ? 0.01 + 0.04 * b / (B - 1) : 0.01;
        numax = fmax(numax, nus[b]);
    }
    const double PI = 3.14159265358979323846, dx = 1.0 / N;
    for (int b = 0; b < B; b++)
        for (int i = 0; i < N; i++) u0[(size_t)b * N + i] = sin(2 * PI * i * dx);

    double dt = 0.4 * fmin(dx, dx * dx / (2 * numax));
    long steps = bench ? bench : (long)ceil(t_end / dt);
    if (!bench) dt = t_end / steps;

    size_t bytes = sizeof(double) * B * N;
    double *u, *un, *s1, *s2, *acc, *dnus;
    CHECK(cudaMalloc(&u, bytes)); CHECK(cudaMalloc(&un, bytes));
    CHECK(cudaMalloc(&s1, bytes)); CHECK(cudaMalloc(&s2, bytes));
    CHECK(cudaMalloc(&acc, bytes)); CHECK(cudaMalloc(&dnus, sizeof(double) * B));
    CHECK(cudaMemcpy(u, u0.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dnus, nus.data(), sizeof(double) * B, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (B * N + threads - 1) / threads;
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (long s = 0; s < steps; s++) {
        stage<<<blocks, threads>>>(u, u,  acc, s1, dnus, N, B, dx, dt / 2, 1.0, 1, 0, dt);
        stage<<<blocks, threads>>>(u, s1, acc, s2, dnus, N, B, dx, dt / 2, 2.0, 0, 0, dt);
        stage<<<blocks, threads>>>(u, s2, acc, s1, dnus, N, B, dx, dt,     2.0, 0, 0, dt);
        stage<<<blocks, threads>>>(u, s1, acc, un, dnus, N, B, dx, 0.0,    1.0, 0, 1, dt);
        double *tmp = u; u = un; un = tmp;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, t0, t1));

    printf("B=%d N=%d steps=%ld dt=%.6e seconds=%.4f updates_per_s=%.3e\n",
           B, N, steps, dt, ms / 1e3, (double)B * N * steps / (ms / 1e3));

    if (!bench) {
        std::vector<double> h((size_t)B * N);
        CHECK(cudaMemcpy(h.data(), u, bytes, cudaMemcpyDeviceToHost));
        FILE *f = fopen("burgers_out.csv", "w");
        if (!f) { perror("burgers_out.csv"); return 1; }
        fprintf(f, "x,u_nu%.4f,u_nu%.4f\n", nus[0], nus[B - 1]);
        for (int i = 0; i < N; i++)
            fprintf(f, "%.17g,%.17g,%.17g\n", i * dx, h[i], h[(size_t)(B - 1) * N + i]);
        fclose(f);
        printf("wrote burgers_out.csv\n");
    }
    cudaFree(u); cudaFree(un); cudaFree(s1); cudaFree(s2); cudaFree(acc); cudaFree(dnus);
    return 0;
}
