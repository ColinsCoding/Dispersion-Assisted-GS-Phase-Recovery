// main.cpp -- the droid CLI: a PTZ camera droid tracks a foam-sword fighter and tags them with a game laser.
//
//   droid [--frames N] [--seed S] [--isolation on|off] [--vision auto|cpu|gpu] [--ascii K] [--csv PATH|-]
//
// Exit codes (read them with $? on Linux/WSL, %ERRORLEVEL% on Windows):
//   0 ok   2 bad command-line arguments   3 GPU vision requested but unavailable   4 cannot write the CSV
#include "droid.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace droid;

namespace {

struct Options {
    int frames = 300;
    unsigned seed = 7;
    bool isolation = true;
    std::string vision = "auto";
    int ascii_every = 30;
    std::string csv = "droid_run.csv";
};

void usage() {
    std::cout << "droid -- Jalali-themed PTZ camera droid (PST vision, motor ODEs, isolated head, Gaussian laser)\n"
                 "  --frames N          frames at 30 fps (default 300)\n"
                 "  --seed S            noise seed (default 7)\n"
                 "  --isolation on|off  vibration-isolated camera head (default on)\n"
                 "  --vision auto|cpu|gpu   PST backend (default auto: GPU if built with CUDA, else CPU)\n"
                 "  --ascii K           print a top-down frame every K frames (0 = off, default 30)\n"
                 "  --csv PATH|-        per-frame log (default droid_run.csv, '-' = none)\n"
                 "exit codes: 0 ok, 2 bad args, 3 GPU unavailable, 4 CSV write failed\n";
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](std::string& v) { if (i + 1 >= argc) return false; v = argv[++i]; return true; };
        std::string v;
        try {
            if (a == "--help" || a == "-h") { usage(); std::exit(0); }
            else if (a == "--frames" && need(v)) { o.frames = std::stoi(v); if (o.frames < 1) return false; }
            else if (a == "--seed" && need(v)) o.seed = static_cast<unsigned>(std::stoul(v));
            else if (a == "--isolation" && need(v)) { if (v != "on" && v != "off") return false; o.isolation = v == "on"; }
            else if (a == "--vision" && need(v)) { if (v != "auto" && v != "cpu" && v != "gpu") return false; o.vision = v; }
            else if (a == "--ascii" && need(v)) { o.ascii_every = std::stoi(v); if (o.ascii_every < 0) return false; }
            else if (a == "--csv" && need(v)) o.csv = v;
            else return false;
        } catch (...) { return false; }
    }
    return true;
}

void draw_ascii(double t, double px, double py, double pan, double range, bool firing, bool tag, bool blocked,
                double err_deg, double fov_deg, double sword_phase) {
    const int W = 61, H = 25;
    const double span = 9.0;                               // metres from centre to edge
    std::string grid(static_cast<size_t>(W) * H, ' ');
    auto put = [&](double x, double y, char c) {
        int cx = static_cast<int>(std::lround((x / span + 1) * 0.5 * (W - 1)));
        int cy = static_cast<int>(std::lround((1 - (y / span + 1) * 0.5) * (H - 1)));
        if (cx >= 0 && cx < W && cy >= 0 && cy < H) grid[static_cast<size_t>(cy) * W + cx] = c;
    };
    for (double s = 0.4; s < range; s += 0.3) put(s * std::cos(pan), s * std::sin(pan), firing ? '*' : '.');
    put(px + 0.7 * std::cos(sword_phase), py + 0.7 * std::sin(sword_phase), '/');
    put(px, py, tag ? 'X' : (blocked ? 'B' : 'P'));
    put(0, 0, 'D');
    std::printf("t=%5.2fs  aim error %5.2f deg  fov %4.1f deg  %s\n", t, err_deg, fov_deg,
                tag ? "TAGGED" : (blocked ? "BLOCKED by foam sword" : (firing ? "laser on" : "tracking")));
    for (int r = 0; r < H; ++r) std::printf("|%s|\n", grid.substr(static_cast<size_t>(r) * W, W).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) { usage(); return 2; }

#ifdef DROID_CUDA
    const bool have_gpu_build = true;
#else
    const bool have_gpu_build = false;
#endif
    if (o.vision == "gpu" && !have_gpu_build) { std::cerr << "this build has no CUDA; rebuild with nvcc\n"; return 3; }
    bool use_gpu = (o.vision == "gpu") || (o.vision == "auto" && have_gpu_build);

    std::ofstream csv;
    if (o.csv != "-") {
        csv.open(o.csv);
        if (!csv) { std::cerr << "cannot write " << o.csv << "\n"; return 4; }
        csv << "frame,t,player_x,player_y,range_m,pan_true_deg,tilt_true_deg,pan_deg,tilt_deg,err_deg,"
               "jitter_mrad,fov_deg,found,firing,tag,blocked,spot_offset_mm,beam_radius_mm,i_pan_A,i_tilt_A,vision_ms\n";
    }

    // ---- scene and hardware constants (game values, labelled as such)
    const double fps = 30.0, frame_dt = 1.0 / fps, sim_dt = 1e-4;
    const Vec3 camera{0, 0, 1.5};
    const double body_r = 0.25, body_size = 0.5;                // player body radius / angular-size basis (m)
    const double rock_A = 2e-3, rock_f = 12.0;                  // droid's own gait rocks the base: 2 mrad at 12 Hz
    const double w0 = 1e-3, lambda = 532e-9;                    // laser: 1 mm waist, 532 nm (game laser)
    const double fov_wide = 40 * PI / 180, fov_tele = 10 * PI / 180;
    const int n = 128;
    const double pst_S = 0.4, pst_W = 20.0, pst_sigma = 0.05;   // PhyCV-style PST; tight low-pass suppresses the sword
    const double locate_thresh = 0.3;

    MotorParams mp;
    MotorState pan_m, tilt_m;
    MountParams mount;
    MountState head;
    double fov = fov_wide;
    int tags = 0, blocks = 0, found_frames = 0, firing_frames = 0;
    double sum_err = 0, sum_jit2 = 0, i_peak = 0, vis_ms_total = 0;
    double t = 0;

    for (int f = 0; f < o.frames; ++f, t += frame_dt) {
        // player: circles the droid at 5-7 m, weaving; swings the foam sword
        double R = 6.0 + 1.0 * std::sin(0.3 * t), psi = 0.45 * t + 0.3 * std::sin(1.1 * t);
        Vec3 P{R * std::cos(psi), R * std::sin(psi), 1.0};
        double sword_phase = 2.0 * t;
        bool guarding = std::fabs(std::sin(sword_phase)) > 0.9;   // sword held across the body

        double pan_true, tilt_true;
        pan_tilt_of(P - camera, pan_true, tilt_true);
        double range = norm(P - camera);
        double jitter = o.isolation ? head.x : rock_A * std::sin(2 * PI * rock_f * t);
        double e_pan = wrap_pi(pan_true - pan_m.theta), e_tilt = tilt_true - (tilt_m.theta + jitter);
        double err = std::hypot(e_pan * std::cos(tilt_true), e_tilt);

        // camera frame -> PST edges -> target centroid -> measured pointing error
        Image img = render_scene(n, e_pan, e_tilt, fov, body_size / range, sword_phase, o.seed + 977u * f);
        std::vector<float> edge;
        auto t0 = std::chrono::steady_clock::now();
        if (use_gpu) {
#ifdef DROID_CUDA
            std::string why;
            if (!pst_gpu(img, pst_S, pst_W, pst_sigma, edge, why)) {
                if (o.vision == "gpu") { std::cerr << "GPU vision failed: " << why << "\n"; return 3; }
                use_gpu = false;
                edge = pst_cpu(img, pst_S, pst_W, pst_sigma);
            }
#endif
        } else {
            edge = pst_cpu(img, pst_S, pst_W, pst_sigma);
        }
        double vis_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        vis_ms_total += vis_ms;
        double cx, cy, ppr = n / fov;
        bool found = locate_target(edge, n, locate_thresh, cx, cy);
        double pan_ref = pan_m.theta, tilt_ref = tilt_m.theta;
        if (found) {
            ++found_frames;
            pan_ref = pan_m.theta + (cx - n / 2.0) / ppr / std::cos(tilt_true);
            tilt_ref = tilt_m.theta - (cy - n / 2.0) / ppr;
        }
        fov = (found && err < 1.0 * PI / 180) ? fov_tele : fov_wide;   // the Z in PTZ: zoom in once locked

        // laser: fire when locked; spot offset at the target from the true aim error
        bool firing = err < 0.5 * PI / 180;
        double spot = range * err, beam = beam_radius(w0, lambda, range);
        bool tag = firing && !guarding && spot < body_r;
        bool blocked = firing && guarding && spot < body_r;
        tags += tag;
        blocks += blocked;
        firing_frames += firing;
        sum_err += err;
        sum_jit2 += (range * jitter) * (range * jitter);

        // physics: both gimbal motors (RK4) and the isolation mount, in 0.1 ms steps through the frame
        int steps = static_cast<int>(std::lround(frame_dt / sim_dt));
        for (int s = 0; s < steps; ++s) {
            pan_m = motor_step(mp, pan_m, pan_ref, sim_dt);
            tilt_m = motor_step(mp, tilt_m, tilt_ref, sim_dt);
            head = mount_step(mount, head, t + s * sim_dt, rock_A, rock_f, sim_dt);
            i_peak = std::max(i_peak, std::max(std::fabs(pan_m.i), std::fabs(tilt_m.i)));
        }

        if (csv.is_open())
            csv << f << ',' << t << ',' << P.x << ',' << P.y << ',' << range << ',' << pan_true * 180 / PI << ','
                << tilt_true * 180 / PI << ',' << pan_m.theta * 180 / PI << ',' << tilt_m.theta * 180 / PI << ','
                << err * 180 / PI << ',' << jitter * 1e3 << ',' << fov * 180 / PI << ',' << found << ',' << firing
                << ',' << tag << ',' << blocked << ',' << spot * 1e3 << ',' << beam * 1e3 << ',' << pan_m.i << ','
                << tilt_m.i << ',' << vis_ms << '\n';
        if (o.ascii_every > 0 && f % o.ascii_every == 0)
            draw_ascii(t, P.x, P.y, pan_m.theta, range, firing, tag, blocked, err * 180 / PI, fov * 180 / PI, psi + sword_phase);
    }
    if (csv.is_open()) {
        csv.close();
        if (!csv) { std::cerr << "error while writing " << o.csv << "\n"; return 4; }
    }

    std::printf("\n=== summary (%d frames, %.1f s) ===\n", o.frames, o.frames / fps);
    std::printf("vision backend      : %s (PST, %dx%d), %.3f ms/frame average\n", use_gpu ? "GPU cuFFT" : "CPU FFT", n, n, vis_ms_total / o.frames);
    std::printf("target found        : %d / %d frames\n", found_frames, o.frames);
    std::printf("mean aim error      : %.3f deg\n", sum_err / o.frames * 180 / PI);
    std::printf("laser on            : %d frames; tags %d, blocked by foam sword %d\n", firing_frames, tags, blocks);
    std::printf("head isolation      : %s; RMS laser-spot jitter at the target %.2f mm\n", o.isolation ? "on" : "off",
                std::sqrt(sum_jit2 / o.frames) * 1e3);
    std::printf("gimbal motor        : peak current %.3f A (R = %.1f ohm, L = %.1f mH)\n", i_peak, mp.R, mp.L * 1e3);
    std::printf("laser spot radius   : %.2f mm at 6 m (w0 = 1 mm, 532 nm, z_R = %.1f m)\n",
                beam_radius(w0, lambda, 6.0) * 1e3, rayleigh_range(w0, lambda));
    if (o.csv != "-") std::printf("per-frame log       : %s\n", o.csv.c_str());
    return 0;
}
