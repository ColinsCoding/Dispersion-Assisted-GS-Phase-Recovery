// test_droid.cpp -- checks every piece against a closed form. Exit code = number of failed checks.
#include "droid.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

using namespace droid;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what, double got = 0, double want = 0) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL  %s  (got %.6g, want %.6g)\n", what, got, want); }
    else std::printf("ok    %s\n", what);
}

int main() {
    // unit vectors
    Vec3 d = aim_vector(0.7, -0.2);
    check(std::fabs(norm(d) - 1) < 1e-15, "aim_vector is a unit vector", norm(d), 1);
    double p, t;
    pan_tilt_of(d, p, t);
    check(std::fabs(p - 0.7) < 1e-14 && std::fabs(t + 0.2) < 1e-14, "pan_tilt_of inverts aim_vector");
    bool threw = false;
    try { unit({0, 0, 0}); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unit(0) refuses: a zero vector has no direction");
    check(std::fabs(wrap_pi(3 * PI) - PI) < 1e-12 || std::fabs(wrap_pi(3 * PI) + PI) < 1e-12, "wrap_pi(3 pi) = +-pi");

    // gimbal motor: settles on the setpoint, and RK4 has converged at the game's 0.1 ms step
    MotorParams mp;
    MotorState s1 = motor_advance(mp, MotorState{}, 0.5, 3.0, 1e-4);
    check(std::fabs(s1.theta - 0.5) < 1e-3, "motor settles to a 0.5 rad step within 3 s", s1.theta, 0.5);
    MotorState s2 = motor_advance(mp, MotorState{}, 0.5, 0.3, 1e-4);
    MotorState s3 = motor_advance(mp, MotorState{}, 0.5, 0.3, 5e-5);
    check(std::fabs(s2.theta - s3.theta) < 1e-8, "RK4 converged: dt vs dt/2 agree to 1e-8 rad", std::fabs(s2.theta - s3.theta), 0);
    check(std::fabs(s1.i) < 1e-2, "motor current decays to ~0 at rest (only friction torque)", s1.i, 0);

    // isolation mount: simulated steady-state amplitude matches the transmissibility formula
    MountParams mt;
    MountState m;
    double A = 2e-3, f = 12.0, dt = 1e-4, peak = 0;
    for (int k = 0; k < 200000; ++k) {                       // 20 s; keep the max over the last 2 s
        m = mount_step(mt, m, k * dt, A, f, dt);
        if (k > 180000) peak = std::max(peak, std::fabs(m.x));
    }
    double T = transmissibility(f / mt.f0, mt.zeta);
    check(std::fabs(peak / A - T) / T < 0.02, "isolation: ODE amplitude matches |X/U| within 2%", peak / A, T);
    check(std::fabs(transmissibility(std::sqrt(2.0), 0.3) - 1) < 1e-12, "transmissibility = 1 at r = sqrt 2 for any damping");

    // Gaussian beam
    double zr = rayleigh_range(1e-3, 532e-9);
    check(std::fabs(beam_radius(1e-3, 532e-9, zr) - std::sqrt(2.0) * 1e-3) < 1e-15, "beam radius = sqrt2 w0 at the Rayleigh range");

    // FFT
    int n = 64;
    std::vector<std::complex<double>> a(n * n), b;
    for (int i = 0; i < n * n; ++i) a[i] = std::complex<double>(std::sin(0.37 * i), std::cos(0.11 * i));
    b = a;
    fft2_cpu(b, n, false);
    fft2_cpu(b, n, true);
    double err = 0;
    for (int i = 0; i < n * n; ++i) err = std::max(err, std::abs(a[i] - b[i]));
    check(err < 1e-12, "FFT forward + inverse round trip < 1e-12", err, 0);
    std::vector<std::complex<double>> delta(n * n, 0.0);
    delta[0] = 1.0;
    fft2_cpu(delta, n, false);
    double dev = 0;
    for (auto& z : delta) dev = std::max(dev, std::abs(z - 1.0));
    check(dev < 1e-12, "FFT of a delta is flat", dev, 0);

    // vision: PST + centroid recovers a known pointing error
    double fov = 40 * PI / 180, e_pan = 4 * PI / 180, e_tilt = -3 * PI / 180;
    Image img = render_scene(128, e_pan, e_tilt, fov, 0.5 / 6.0, PI / 2, 11);
    std::vector<float> edge = pst_cpu(img, 0.4, 20.0, 0.05);
    double cx, cy;
    bool found = locate_target(edge, 128, 0.3, cx, cy);
    double want_x = 64 + e_pan * 128 / fov, want_y = 64 - e_tilt * 128 / fov;
    double miss_deg = std::hypot(cx - want_x, cy - want_y) / (128 / fov) * 180 / PI;
    check(found && miss_deg < 0.5, "PST locates the target within 0.5 deg (sword in frame)", miss_deg, 0);
    // target fully in frame: accurate for every sword angle at both zooms
    double worst = 0;
    bool right_way = true;
    for (int k = 0; k < 8; ++k)
        for (double fv : {40.0, 10.0}) {
            double fr = fv * PI / 180, ep = 0.12 * fr * std::cos(k), et = -0.1 * fr * std::sin(k);
            Image im = render_scene(128, ep, et, fr, 0.5 / 6.0, k * PI / 4, 31 + k);
            double ax, ay;
            if (!locate_target(pst_cpu(im, 0.4, 20.0, 0.05), 128, 0.3, ax, ay)) { worst = 99; continue; }
            worst = std::max(worst, std::hypot(ax - 64 - ep * 128 / fr, ay - 64 + et * 128 / fr) / (128 / fr) * 180 / PI);
            // target partly out of frame (0.3 fov off-centre): biased toward the centre, but must still point
            // the right way so the tracking loop recentres it
            double bp = 0.3 * fr * (k % 2 ? 1 : -1);
            Image edge_im = render_scene(128, bp, 0.0, fr, 0.5 / 6.0, k * PI / 4, 51 + k);
            if (!locate_target(pst_cpu(edge_im, 0.4, 20.0, 0.05), 128, 0.3, ax, ay) || (ax - 64) * bp <= 0) right_way = false;
        }
    check(worst < 0.5, "locator, target in frame: worst over 8 sword angles x 2 zooms < 0.5 deg", worst, 0);
    check(right_way, "locator, target half out of frame: still points the right way (tracker recentres)");

#ifdef DROID_CUDA
    std::vector<float> g;
    std::string why;
    bool ok = pst_gpu(img, 0.4, 20.0, 0.05, g, why);
    double gd = 0;
    if (ok) for (size_t i = 0; i < g.size(); ++i) {
        double diff = std::fabs(g[i] - edge[i]);
        diff = std::min(diff, 2 * PI - diff);                 // phases near +-pi are the same angle
        gd = std::max(gd, diff);
    }
    check(ok && gd < 2e-3, "GPU (cuFFT, float) PST matches CPU (double) within 2e-3 rad", gd, 0);
#else
    std::printf("skip  GPU check (built without CUDA)\n");
#endif

    std::printf("\n%d / %d checks passed\n", checks - failures, checks);
    return failures;
}
