// render3d.cu -- the droid's camera view on the GPU: one thread per pixel runs the same HD ray tracer as
// the CPU (scene3d.hpp). The device buffer is allocated once and reused across frames (the lesson from
// the 2D version: per-frame allocation, not compute, dominated GPU time).
#include "game3d.hpp"

#include <cuda_runtime.h>

namespace g3 {

namespace {
__global__ void render_kernel(d3::Scene s, d3::Camera c, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= c.n || j >= c.n) return;
    d3::V3 p = d3::render_pixel(s, c, i, j);
    size_t k = (static_cast<size_t>(j) * c.n + i) * 3;
    out[k] = p.x; out[k + 1] = p.y; out[k + 2] = p.z;
}

float* g_buf = nullptr;
size_t g_cap = 0;
}  // namespace

bool render_gpu(const d3::Scene& s, const d3::Camera& c, std::vector<float>& rgb, std::string& err) {
    size_t count = static_cast<size_t>(c.n) * c.n * 3;
    if (count > g_cap) {
        if (g_buf) cudaFree(g_buf);
        g_buf = nullptr;
        g_cap = 0;
        if (cudaMalloc(&g_buf, count * sizeof(float)) != cudaSuccess) { err = "cudaMalloc failed (is there a CUDA GPU?)"; return false; }
        g_cap = count;
    }
    dim3 block(16, 16), grid((c.n + 15) / 16, (c.n + 15) / 16);
    render_kernel<<<grid, block>>>(s, c, g_buf);
    if (cudaGetLastError() != cudaSuccess) { err = "render kernel launch failed"; return false; }
    rgb.resize(count);
    if (cudaMemcpy(rgb.data(), g_buf, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) { err = "copy from GPU failed"; return false; }
    return true;
}

}  // namespace g3
