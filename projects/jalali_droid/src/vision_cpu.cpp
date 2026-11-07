// vision_cpu.cpp -- what the droid's camera sees, and how it finds you: a synthetic frame, a radix-2 FFT,
// and the Phase Stretch Transform (PST) from Jalali's lab: low-pass the spectrum, multiply by a nonlinear
// "warped" phase exp(-i S phi(r)), transform back, and read the PHASE -- it peaks at edges.
#include "droid.hpp"

#include <algorithm>
#include <stdexcept>

namespace droid {

namespace {
// small deterministic generator (same numbers on every compiler / OS)
struct Lcg {
    unsigned s;
    explicit Lcg(unsigned seed) : s(seed ? seed : 1u) {}
    double uniform() { s = 1664525u * s + 1013904223u; return (s >> 8) * (1.0 / 16777216.0); }
};

bool is_pow2(int n) { return n > 1 && (n & (n - 1)) == 0; }

void fft1(std::complex<double>* a, int n, int stride, bool inverse) {
    for (int i = 1, j = 0; i < n; ++i) {                       // bit-reversal permutation
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i * stride], a[j * stride]);
    }
    for (int len = 2; len <= n; len <<= 1) {                   // butterflies
        double ang = 2 * PI / len * (inverse ? 1 : -1);
        std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1, 0);
            for (int k = 0; k < len / 2; ++k) {
                std::complex<double> u = a[(i + k) * stride], v = a[(i + k + len / 2) * stride] * w;
                a[(i + k) * stride] = u + v;
                a[(i + k + len / 2) * stride] = u - v;
                w *= wl;
            }
        }
    }
}

double freq(int k, int n) { return (k < n / 2 ? k : k - n) / static_cast<double>(n); }
}  // namespace

Image render_scene(int n, double err_pan, double err_tilt, double fov, double target_size, double sword_phase, unsigned seed) {
    if (!is_pow2(n)) throw std::invalid_argument("render_scene: n must be a power of two");
    if (!(fov > 0)) throw std::invalid_argument("render_scene: fov must be positive");
    Image img;
    img.n = n;
    img.px.assign(static_cast<size_t>(n) * n, 0.f);
    Lcg rng(seed);
    double c = n / 2.0, ppr = n / fov;                         // pixels per radian
    double cx = c + err_pan * ppr, cy = c - err_tilt * ppr;     // target center (image y points down)
    double rpx = std::max(1.5, 0.5 * target_size * ppr);       // target radius in pixels
    double sx = std::cos(sword_phase), sy = -std::sin(sword_phase);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            double v = 0.2 + 0.05 * std::sin(2 * PI * x / 37.0) * std::cos(2 * PI * y / 53.0);
            double d = std::hypot(x - cx, y - cy);
            double body = 0.5 * (1 - std::tanh((d - rpx) / 0.8));            // soft-edged disk
            // foam sword: a bright segment from the body's edge, swinging with sword_phase
            double px = x - (cx + rpx * sx), py = y - (cy + rpx * sy);
            double along = px * sx + py * sy, across = std::fabs(-px * sy + py * sx);
            double sword = (along >= 0 && along <= 1.5 * rpx && across <= 1.0) ? 1.0 : 0.0;
            v = std::max(v, 0.2 + 0.7 * body);
            v = std::max(v, sword * 0.95);
            v += 0.02 * (rng.uniform() - 0.5);
            img.px[static_cast<size_t>(y) * n + x] = static_cast<float>(std::min(1.0, std::max(0.0, v)));
        }
    }
    return img;
}

void fft2_cpu(std::vector<std::complex<double>>& a, int n, bool inverse) {
    if (!is_pow2(n) || a.size() != static_cast<size_t>(n) * n) throw std::invalid_argument("fft2_cpu: need n x n, n a power of two");
    for (int r = 0; r < n; ++r) fft1(&a[static_cast<size_t>(r) * n], n, 1, inverse);
    for (int c = 0; c < n; ++c) fft1(&a[c], n, n, inverse);
    if (inverse) for (auto& z : a) z /= static_cast<double>(n) * n;
}

std::vector<double> pst_kernel(int n, double S, double W, double sigma_lp, std::vector<double>& lowpass) {
    std::vector<double> k(static_cast<size_t>(n) * n);
    lowpass.assign(k.size(), 0.0);
    double kmax = 0;
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            double r = std::hypot(freq(x, n), freq(y, n)), wr = W * r;
            size_t i = static_cast<size_t>(y) * n + x;
            k[i] = wr * std::atan(wr) - 0.5 * std::log1p(wr * wr);   // PhyCV's warped phase kernel
            kmax = std::max(kmax, k[i]);
            lowpass[i] = std::exp(-(r / sigma_lp) * (r / sigma_lp));
        }
    for (auto& v : k) v = S * v / kmax;
    return k;
}

std::vector<float> pst_cpu(const Image& img, double S, double W, double sigma_lp) {
    int n = img.n;
    std::vector<std::complex<double>> a(img.px.begin(), img.px.end());
    fft2_cpu(a, n, false);
    std::vector<double> lp;
    std::vector<double> k = pst_kernel(n, S, W, sigma_lp, lp);
    for (size_t i = 0; i < a.size(); ++i) a[i] *= lp[i] * std::polar(1.0, -k[i]);
    fft2_cpu(a, n, true);
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = static_cast<float>(std::arg(a[i]));
    return out;
}

bool locate_target(const std::vector<float>& edge, int n, double thresh, double& cx, double& cy) {
    // Weighted centroid of the POSITIVE PST phase above thresh * max. Chosen by a sweep over 60 scenes
    // (sword angles, offsets, both zooms): with a tight PST low-pass (sigma ~ 0.05) the thin foam sword's
    // edges are suppressed and the positive phase rings the body symmetrically -- mean error 0.18 deg,
    // worst 0.35 deg, vs 1-2 deg for |phase| or median variants at sigma = 0.15.
    double mx = 0;
    for (float v : edge) mx = std::max(mx, static_cast<double>(v));
    if (mx < 1e-6) return false;
    double sw = 0, sx = 0, sy = 0;
    int cnt = 0;
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            double w = edge[static_cast<size_t>(y) * n + x];
            if (w >= thresh * mx) { sw += w; sx += w * x; sy += w * y; ++cnt; }
        }
    if (cnt < 5) return false;
    cx = sx / sw;
    cy = sy / sw;
    return true;
}

}  // namespace droid
