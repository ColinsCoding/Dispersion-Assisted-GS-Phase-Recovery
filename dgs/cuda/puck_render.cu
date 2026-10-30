// puck_render.cu - ray-cast an air hockey puck (a solid cylinder) on the GPU, one thread per pixel.
//
//   puck_render out.ppm width height tilt_deg spin_deg h_over_R
//
// Linear algebra: the puck's orientation is the rotation matrix M = Rx(tilt) Rz(spin). Each
// orthographic camera ray (origin (u, v, 10), direction (0, 0, -1)) is taken into the puck's own
// frame with M^T (the inverse of a rotation is its transpose), intersected with the cylinder
// x^2 + y^2 <= 1, |z| <= h/2 (caps and side), and the local normal is rotated back with M for
// Lambert shading. The top face carries a red stripe so the spin angle is visible.
// Prints "kernel_ms=<t> puck_pixels=<n>".
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

struct Mat3 { float m[9]; };

__host__ __device__ inline void mul(const Mat3 &A, const float v[3], float out[3]) {
    for (int i = 0; i < 3; i++) out[i] = A.m[3 * i] * v[0] + A.m[3 * i + 1] * v[1] + A.m[3 * i + 2] * v[2];
}
__host__ __device__ inline void mulT(const Mat3 &A, const float v[3], float out[3]) {
    for (int i = 0; i < 3; i++) out[i] = A.m[i] * v[0] + A.m[3 + i] * v[1] + A.m[6 + i] * v[2];
}

static Mat3 rotation(float tilt, float spin) {
    float ct = cosf(tilt), st = sinf(tilt), cs = cosf(spin), ss = sinf(spin);
    // Rx(tilt) * Rz(spin)
    Mat3 M = {{cs, -ss, 0.f,
               ct * ss, ct * cs, -st,
               st * ss, st * cs, ct}};
    return M;
}

__global__ void render(unsigned char *img, int W, int H, Mat3 M, float half_h, float extent, int *count) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    float u = (2.f * (px + 0.5f) / W - 1.f) * extent, v = (1.f - 2.f * (py + 0.5f) / H) * extent;
    float ow[3] = {u, v, 10.f}, dw[3] = {0.f, 0.f, -1.f}, o[3], d[3];
    mulT(M, ow, o);
    mulT(M, dw, d);

    float tbest = 1e30f, n[3] = {0, 0, 0};
    // caps z = +-half_h
    if (fabsf(d[2]) > 1e-6f) {
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float t = (sgn * half_h - o[2]) / d[2];
            float x = o[0] + t * d[0], y = o[1] + t * d[1];
            if (t > 0.f && t < tbest && x * x + y * y <= 1.f) { tbest = t; n[0] = 0; n[1] = 0; n[2] = (float)sgn; }
        }
    }
    // side x^2 + y^2 = 1
    float a = d[0] * d[0] + d[1] * d[1];
    if (a > 1e-8f) {
        float b = 2.f * (o[0] * d[0] + o[1] * d[1]), c = o[0] * o[0] + o[1] * o[1] - 1.f, disc = b * b - 4.f * a * c;
        if (disc >= 0.f) {
            float t = (-b - sqrtf(disc)) / (2.f * a);
            float z = o[2] + t * d[2];
            if (t > 0.f && t < tbest && fabsf(z) <= half_h) {
                tbest = t; n[0] = o[0] + t * d[0]; n[1] = o[1] + t * d[1]; n[2] = 0.f;
            }
        }
    }
    unsigned char *p = img + 3 * (py * W + px);
    if (tbest == 1e30f) { p[0] = 200; p[1] = 225; p[2] = 245; return; }   // table: pale blue
    atomicAdd(count, 1);
    float nw[3], light[3] = {-0.35f, 0.45f, 0.82f};
    mul(M, n, nw);
    float lam = fmaxf(0.f, nw[0] * light[0] + nw[1] * light[1] + nw[2] * light[2]);
    float shade = 0.18f + 0.82f * lam, r = 0.12f, g = 0.12f, bl = 0.14f;              // black rubber
    float hx = o[0] + tbest * d[0], hy = o[1] + tbest * d[1];
    if (n[2] > 0.5f && fabsf(hy) < 0.12f && hx > -0.1f) { r = 0.85f; g = 0.1f; bl = 0.1f; }   // stripe
    p[0] = (unsigned char)fminf(255.f, 255.f * r * shade * 2.2f);
    p[1] = (unsigned char)fminf(255.f, 255.f * g * shade * 2.2f);
    p[2] = (unsigned char)fminf(255.f, 255.f * bl * shade * 2.2f);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s out.ppm width height tilt_deg spin_deg h_over_R\n", argv[0]);
        return 2;
    }
    int W = atoi(argv[2]), H = atoi(argv[3]);
    float tilt = atof(argv[4]) * 3.14159265f / 180.f, spin = atof(argv[5]) * 3.14159265f / 180.f;
    float hR = atof(argv[6]);
    if (W < 8 || H < 8 || W > 8192 || H > 8192 || hR <= 0.f) {
        fprintf(stderr, "need 8 <= width, height <= 8192 and h_over_R > 0\n");
        return 2;
    }
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { fprintf(stderr, "no CUDA device\n"); return 3; }

    Mat3 M = rotation(tilt, spin);
    unsigned char *d_img; int *d_count, count = 0;
    cudaMalloc(&d_img, (size_t)W * H * 3);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    render<<<grd, blk>>>(d_img, W, H, M, 0.5f * hR, 1.6f, d_count);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }

    std::vector<unsigned char> img((size_t)W * H * 3);
    cudaMemcpy(img.data(), d_img, img.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    FILE *f = fopen(argv[1], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[1]); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(img.data(), 1, img.size(), f);
    fclose(f);
    cudaFree(d_img); cudaFree(d_count);
    printf("kernel_ms=%.4f puck_pixels=%d\n", ms, count);
    return 0;
}
