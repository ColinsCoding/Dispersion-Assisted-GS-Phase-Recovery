// main3d.cpp -- the 3D battle: the Jalali Guardian (PTZ camera droid) vs. Souma the Foam Swordsman on a
// rolling-hill field. Each frame: ray-trace the droid's camera view (GPU or CPU) -> orange-ness feature ->
// Phase Stretch Transform -> target bearing -> motor ODEs (RK4) with velocity feed-forward -> 3D laser ray
// (body hit / sword parry / blocked by a hill) -> monster-battle damage from assets/characters.def.
//
//   droid3d [--frames N] [--isolation on|off] [--render auto|cpu|gpu] [--vision auto|cpu|gpu] [--n PX]
//           [--defs PATH] [--csv PATH|-] [--frames-dir DIR] [--frame-every K] [--dump-frame F --dump PATH]
//
// Exit codes ($? / %ERRORLEVEL%): 0 ok, 2 bad args, 3 GPU requested but unavailable, 4 cannot write output,
// 5 definitions file missing or invalid.
#include "droid.hpp"
#include "game3d.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using d3::V3;
using droid::PI;

namespace {

struct Opt {
    int frames = 600, n = 128, frame_every = 3, dump_frame = -1;
    bool isolation = true;
    std::string render = "auto", vision = "auto", defs = "assets/characters.def", csv = "out/droid3d_run.csv";
    std::string frames_dir, dump = "out/frame.bin";
};

void usage() {
    std::cout << "droid3d -- 3D battle: Jalali Guardian vs. Souma the Foam Swordsman\n"
                 "  --frames N  --isolation on|off  --render auto|cpu|gpu  --vision auto|cpu|gpu  --n PX (power of 2)\n"
                 "  --defs PATH  --csv PATH|-  --frames-dir DIR  --frame-every K  --dump-frame F --dump PATH\n"
                 "exit codes: 0 ok, 2 bad args, 3 GPU unavailable, 4 cannot write output, 5 bad definitions\n";
}

bool parse(int argc, char** argv, Opt& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i], v;
        auto need = [&]() { if (i + 1 >= argc) return false; v = argv[++i]; return true; };
        try {
            if (a == "--help" || a == "-h") { usage(); std::exit(0); }
            else if (a == "--frames" && need()) { o.frames = std::stoi(v); if (o.frames < 1) return false; }
            else if (a == "--n" && need()) { o.n = std::stoi(v); if (o.n < 32 || (o.n & (o.n - 1))) return false; }
            else if (a == "--isolation" && need()) { if (v != "on" && v != "off") return false; o.isolation = v == "on"; }
            else if (a == "--render" && need()) { if (v != "auto" && v != "cpu" && v != "gpu") return false; o.render = v; }
            else if (a == "--vision" && need()) { if (v != "auto" && v != "cpu" && v != "gpu") return false; o.vision = v; }
            else if (a == "--defs" && need()) o.defs = v;
            else if (a == "--csv" && need()) o.csv = v;
            else if (a == "--frames-dir" && need()) o.frames_dir = v;
            else if (a == "--frame-every" && need()) { o.frame_every = std::stoi(v); if (o.frame_every < 1) return false; }
            else if (a == "--dump-frame" && need()) o.dump_frame = std::stoi(v);
            else if (a == "--dump" && need()) o.dump = v;
            else return false;
        } catch (...) { return false; }
    }
    return true;
}

struct Lcg {
    unsigned s;
    explicit Lcg(unsigned seed) : s(seed) {}
    double uniform() { s = 1664525u * s + 1013904223u; return (s >> 8) * (1.0 / 16777216.0); }
};

struct HeroKin { V3 foot; double R, jump; bool dashing; };

HeroKin hero_at(double t, const d3::Terrain& T, double dash_period, double v0, double g) {
    double psi = 0.35 * t + 0.3 * std::sin(1.1 * t);
    double Rb = 8.0 + 1.5 * std::sin(0.3 * t);
    double tau = std::fmod(t, dash_period) - (dash_period - 3.0);   // dash during the last 3 s of each period
    bool dash = tau >= 0;
    double R = Rb - (Rb - 1.3) * (dash ? std::sin(PI * tau / 3.0) : 0.0);
    double tj = std::fmod(t, 3.0), air = 2 * v0 / g;
    double jump = tj < air ? v0 * tj - 0.5 * g * tj * tj : 0.0;     // ballistic hop every 3 s
    float x = static_cast<float>(R * std::cos(psi)), y = static_cast<float>(R * std::sin(psi));
    return {d3::mk(x, y, d3::height(T, x, y) + static_cast<float>(jump)), R, jump, dash};
}

d3::Hero hero_geometry(const HeroKin& k, double t) {
    V3 to_droid = d3::normalize(d3::mk(-k.foot.x, -k.foot.y, 0));
    V3 right = d3::normalize(d3::cross(to_droid, d3::mk(0, 0, 1)));
    d3::Hero h;
    h.body = k.foot + d3::mk(0, 0, 0.95f);
    h.body_r = 0.32f;
    h.head = k.foot + d3::mk(0, 0, 1.55f);
    h.head_r = 0.18f;
    V3 hand = h.body + 0.30f * right + 0.30f * to_droid - d3::mk(0, 0, 0.05f);
    float ph = static_cast<float>(2.2 * t);                          // the foam sword sweeps across the body
    V3 dir = d3::normalize(std::cos(ph) * (-1.0f * right) + std::sin(ph) * d3::mk(0, 0, 1));
    h.sword_a = hand;
    h.sword_b = hand + 1.0f * dir;
    h.sword_r = 0.05f;
    return h;
}

bool write_ppm(const std::string& path, const std::vector<float>& rgb, int n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << n << " " << n << "\n255\n";
    for (float v : rgb) f.put(static_cast<char>(static_cast<unsigned char>(std::lround(255.0 * std::min(1.0f, std::max(0.0f, v))))));
    return static_cast<bool>(f);
}

const char* hit_name(int id) {
    switch (id) { case d3::BODY: return "body"; case d3::HEAD: return "head"; case d3::SWORD: return "sword"; case d3::GROUND: return "hill"; default: return "sky"; }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);                                        // show the Japanese names
#endif
    Opt o;
    if (!parse(argc, argv, o)) { usage(); return 2; }
#ifdef DROID_CUDA
    const bool cuda_build = true;
#else
    const bool cuda_build = false;
#endif
    if ((o.render == "gpu" || o.vision == "gpu") && !cuda_build) { std::cerr << "this build has no CUDA\n"; return 3; }
    bool gpu_render = o.render == "gpu" || (o.render == "auto" && cuda_build);
    bool gpu_vision = o.vision == "gpu" || (o.vision == "auto" && cuda_build);

    g3::Defs defs;
    std::string err;
    g3::Fighter guardian, hero;
    double hold_s, cooldown_s, reach, v0, dash_period, g, dni, dhi, eta;
    d3::Scene scene;
    std::string solar_ja;
    try {
        if (!g3::load_defs(o.defs, defs, err)) { std::cerr << err << "\n"; return 5; }
        guardian = g3::fighter_from(defs, "guardian");
        hero = g3::fighter_from(defs, "hero");
        hold_s = g3::num(defs, "guardian", "hit_hold_s");
        cooldown_s = g3::num(defs, "guardian", "cooldown_s");
        reach = g3::num(defs, "hero", "reach_m");
        v0 = g3::num(defs, "hero", "jump_speed_mps");
        dash_period = g3::num(defs, "hero", "dash_period_s");
        g = g3::num(defs, "world", "gravity");
        scene.sun = d3::normalize(d3::mk(static_cast<float>(g3::num(defs, "world", "sun_x")), static_cast<float>(g3::num(defs, "world", "sun_y")),
                                         static_cast<float>(g3::num(defs, "world", "sun_z"))));
        g3::build_solar_field(defs, scene);
        dni = g3::num(defs, "solar", "dni_w_m2");
        dhi = g3::num(defs, "solar", "dhi_w_m2");
        eta = g3::num(defs, "solar", "module_efficiency");
        solar_ja = g3::str(defs, "solar", "name_ja");
    } catch (const std::exception& e) { std::cerr << "definitions: " << e.what() << "\n"; return 5; }

    std::ofstream csv;
    if (o.csv != "-") {
        csv.open(o.csv);
        if (!csv) { std::cerr << "cannot write " << o.csv << "\n"; return 4; }
        csv << "frame,t,hero_x,hero_y,hero_z,jump_m,R_m,dashing,pan_deg,tilt_deg,true_err_deg,found,los_clear,firing,laser_hit,"
               "guardian_hp,hero_hp,event,render_ms,vision_ms,pv_kw,shaded_panels,droid_motor_w\n";
    }

    std::printf("%s (%s)  vs  %s (%s)  on the %s\n", guardian.name_ja.c_str(), guardian.name_en.c_str(), hero.name_ja.c_str(),
                hero.name_en.c_str(), g3::str(defs, "world", "name_en").c_str());
    std::printf("  %s: %s type, HP %d, move %s / %s\n", guardian.name_en.c_str(), guardian.type.c_str(), guardian.hp, guardian.move_ja.c_str(), guardian.move_en.c_str());
    std::printf("  %s: %s type, HP %d, move %s / %s\n", hero.name_en.c_str(), hero.type.c_str(), hero.hp, hero.move_ja.c_str(), hero.move_en.c_str());
    std::printf("  %s: %d panels, %.0f%% modules, tilt %.1f deg (site latitude), sun %.0f deg above the horizon\n\n", solar_ja.c_str(),
                scene.n_panels, eta * 100, g3::num(defs, "solar", "latitude_deg"), std::asin(scene.sun.z) * 180 / PI);

    const double fps = 30.0, dt = 1.0 / fps, sim_dt = 1e-4;
    const double rock_A = 2e-3, rock_f = 12.0;
    const double fov_wide = 45 * PI / 180, fov_tele = 14 * PI / 180;
    const V3 cam_pos = d3::mk(0, 0, d3::height(scene.T, 0, 0) + 2.0f);
    droid::MotorParams mp;
    droid::MotorState pan_m, tilt_m;
    tilt_m.theta = -0.12;
    droid::MountParams mount;
    droid::MountState head;
    double fov = fov_wide, last_pan = 0, last_tilt = 0, rate_pan = 0, rate_tilt = 0, hold = 0, next_g = 0, next_h = 0;
    bool have_last = false;
    int parries = 0, hill_blocks = 0, g_hits = 0, h_hits = 0, found_frames = 0, los_frames = 0;
    double render_ms = 0, vision_ms = 0, sum_err = 0;
    double pv_sum_kw = 0, pv_min_kw = 1e9, pv_max_kw = 0, motor_w_sum = 0;
    int shaded_frames = 0;
    // static baseline with the hero far away: row-to-row self-shading at this sun angle (sun, terrain, panels are fixed)
    double pv_base_w = 0, pv_ideal_w = 0;
    {
        d3::Scene empty = scene;
        empty.hero = hero_geometry({d3::mk(500, 500, 0), 500, 0, false}, 0);
        for (int p = 0; p < empty.n_panels; ++p) {
            double un;
            pv_base_w += g3::panel_power(empty, p, dni, dhi, eta, un);
            const d3::Panel& P = empty.panels[p];
            pv_ideal_w += eta * 4 * P.hu * P.hv * (dni * std::max(0.0f, d3::dot(P.n, empty.sun)) + dhi * 0.5 * (1 + P.n.z));
        }
    }
    Lcg rng(7);
    std::vector<float> rgb;
    std::string winner;
    int f = 0;

    for (; f < o.frames; ++f) {
        double t = f * dt;
        HeroKin k = hero_at(t, scene.T, dash_period, v0, g);
        scene.hero = hero_geometry(k, t);
        double pv_w = 0;
        int shaded_panels = 0;
        for (int p = 0; p < scene.n_panels; ++p) {                    // the solar farm, this frame
            double un;
            pv_w += g3::panel_power(scene, p, dni, dhi, eta, un);
            shaded_panels += un < 1.0;
        }
        pv_sum_kw += pv_w / 1e3;
        pv_min_kw = std::min(pv_min_kw, pv_w / 1e3);
        pv_max_kw = std::max(pv_max_kw, pv_w / 1e3);
        shaded_frames += pv_w < pv_base_w - 1e-6;                   // the hero's shadow is on the farm this frame
        double jitter = o.isolation ? head.x : rock_A * std::sin(2 * PI * rock_f * t);

        d3::Camera cam{cam_pos, static_cast<float>(pan_m.theta), static_cast<float>(tilt_m.theta + jitter), static_cast<float>(fov), o.n};
        auto t0 = std::chrono::steady_clock::now();
        bool rendered = false;
#ifdef DROID_CUDA
        if (gpu_render) {
            rendered = g3::render_gpu(scene, cam, rgb, err);
            if (!rendered) { if (o.render == "gpu") { std::cerr << "GPU render failed: " << err << "\n"; return 3; } gpu_render = false; }
        }
#endif
        if (!rendered) g3::render_cpu(scene, cam, rgb);
        auto t1 = std::chrono::steady_clock::now();
        render_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        droid::Image feat;
        feat.n = o.n;
        feat.px.resize(static_cast<size_t>(o.n) * o.n);
        // pedestal 0.2: PST reads the PHASE of a complex field, which is pure noise where the amplitude is ~0;
        // an all-zero background would scatter random "edges" over the empty image and drag the centroid
        for (size_t p = 0; p < feat.px.size(); ++p)
            feat.px[p] = 0.2f + 0.8f * d3::target_feature(d3::mk(rgb[3 * p], rgb[3 * p + 1], rgb[3 * p + 2]));
        std::vector<float> edge;
        bool did = false;
#ifdef DROID_CUDA
        if (gpu_vision) {
            did = droid::pst_gpu(feat, 0.4, 20.0, 0.05, edge, err);
            if (!did) { if (o.vision == "gpu") { std::cerr << "GPU vision failed: " << err << "\n"; return 3; } gpu_vision = false; }
        }
#endif
        if (!did) edge = droid::pst_cpu(feat, 0.4, 20.0, 0.05);
        double cx, cy;
        bool found = droid::locate_target(edge, o.n, 0.3, cx, cy);
        double frame_vision_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        vision_ms += frame_vision_ms;

        // bearing of the target from the droid's own (motor) angles: jitter it does not know about is noise
        d3::Camera motor_cam{cam_pos, static_cast<float>(pan_m.theta), static_cast<float>(tilt_m.theta), static_cast<float>(fov), o.n};
        V3 fw, rt, up;
        d3::camera_basis(motor_cam, fw, rt, up);
        double true_pan, true_tilt;
        droid::pan_tilt_of({scene.hero.body.x - cam_pos.x, scene.hero.body.y - cam_pos.y, scene.hero.body.z - cam_pos.z}, true_pan, true_tilt);
        V3 tb = d3::normalize(scene.hero.body - cam_pos);
        double true_err = std::acos(std::min(1.0f, d3::dot(tb, fw)));
        bool los = !d3::terrain_blocks(scene.T, cam_pos, scene.hero.body);
        los_frames += los;
        sum_err += true_err;

        double pan_ref, tilt_ref, meas_err = 1e9;
        if (found) {
            ++found_frames;
            float s = std::tan(0.5f * static_cast<float>(fov));
            float px = static_cast<float>((2 * (cx + 0.5) / o.n - 1) * s), py = static_cast<float>((1 - 2 * (cy + 0.5) / o.n) * s);
            V3 d = d3::normalize(fw + px * rt + py * up);
            double mp_pan, mp_tilt;
            droid::pan_tilt_of({d.x, d.y, d.z}, mp_pan, mp_tilt);
            meas_err = std::acos(std::min(1.0f, d3::dot(d, fw)));
            if (have_last) {                                           // alpha-beta style rate estimate
                rate_pan = 0.6 * rate_pan + 0.4 * droid::wrap_pi(mp_pan - last_pan) / dt;
                rate_tilt = 0.6 * rate_tilt + 0.4 * (mp_tilt - last_tilt) / dt;
            }
            last_pan = mp_pan; last_tilt = mp_tilt; have_last = true;
            pan_ref = mp_pan + rate_pan * dt;                          // velocity feed-forward: lead the target
            tilt_ref = mp_tilt + rate_tilt * dt;
            fov = meas_err < 1.0 * PI / 180 ? fov_tele : fov_wide;
        } else {                                                       // lost it (hill or out of frame): sweep
            have_last = false;
            rate_pan = rate_tilt = 0;
            pan_ref = pan_m.theta + 0.05;                              // ~1.5 rad/s search sweep
            tilt_ref = -0.10;
            fov = fov_wide;
        }

        // laser: the droid fires when its own measurement says it is on target
        bool firing = found && meas_err < 0.4 * PI / 180;
        int laser_hit = -1;
        std::string event;
        if (firing) {
            V3 rd = d3::aim(static_cast<float>(pan_m.theta), static_cast<float>(tilt_m.theta + jitter));
            laser_hit = d3::trace(scene, cam_pos, rd, 80.0f).id;
            if (laser_hit == d3::BODY || laser_hit == d3::HEAD) {
                hold += dt;
                if (hold >= hold_s && t >= next_g) {
                    double m = g3::type_multiplier(defs, guardian.move_type, hero.type);
                    int dmg = g3::damage(guardian, hero, m, rng.uniform());
                    hero.hp = std::max(0, hero.hp - dmg);
                    ++g_hits;
                    next_g = t + cooldown_s;
                    hold = 0;
                    event = guardian.move_en + " -" + std::to_string(dmg);
                    std::printf("t=%6.2fs  %s! %s takes %d (x%.1f)  HP %d/%d\n", t, guardian.move_ja.c_str(), hero.name_en.c_str(), dmg, m, hero.hp, hero.max_hp);
                }
            } else {
                hold = 0;
                if (laser_hit == d3::SWORD) { ++parries; event = "parried by foam sword"; }
                if (laser_hit == d3::GROUND) { ++hill_blocks; event = "laser blocked by hill"; }
            }
        } else hold = 0;
        if (k.dashing && k.R < reach && t >= next_h) {                  // the hero dives in and strikes
            double m = g3::type_multiplier(defs, hero.move_type, guardian.type);
            int dmg = g3::damage(hero, guardian, m, rng.uniform());
            guardian.hp = std::max(0, guardian.hp - dmg);
            ++h_hits;
            next_h = t + 2.0;
            event += (event.empty() ? "" : "; ") + hero.move_en + " -" + std::to_string(dmg);
            std::printf("t=%6.2fs  %s! %s takes %d (x%.1f)  HP %d/%d\n", t, hero.move_ja.c_str(), guardian.name_en.c_str(), dmg, m, guardian.hp, guardian.max_hp);
        }

        if (csv.is_open())
            csv << f << ',' << t << ',' << k.foot.x << ',' << k.foot.y << ',' << k.foot.z << ',' << k.jump << ',' << k.R << ',' << k.dashing
                << ',' << pan_m.theta * 180 / PI << ',' << tilt_m.theta * 180 / PI << ',' << true_err * 180 / PI << ',' << found << ','
                << los << ',' << firing << ',' << laser_hit << ',' << guardian.hp << ',' << hero.hp << ",\"" << event << "\","
                << std::chrono::duration<double, std::milli>(t1 - t0).count() << ',' << frame_vision_ms << ',' << pv_w / 1e3 << ','
                << shaded_panels << ',' << (mp.R * (pan_m.i * pan_m.i + tilt_m.i * tilt_m.i) + mp.Ke * std::fabs(pan_m.omega * pan_m.i + tilt_m.omega * tilt_m.i)) << '\n';
        if (!o.frames_dir.empty() && f % o.frame_every == 0) {
            char name[64];
            std::snprintf(name, sizeof name, "/frame_%05d.ppm", f);
            if (!write_ppm(o.frames_dir + name, rgb, o.n)) { std::cerr << "cannot write frames to " << o.frames_dir << "\n"; return 4; }
        }
        if (f == o.dump_frame) {
            std::ofstream b(o.dump, std::ios::binary), s(o.dump + ".scene.txt");
            if (!b || !s) { std::cerr << "cannot write " << o.dump << "\n"; return 4; }
            b.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size() * sizeof(float)));
            const d3::Hero& h = scene.hero;
            s << "n=" << o.n << "\ncam=" << cam.pos.x << "," << cam.pos.y << "," << cam.pos.z << "\npan=" << cam.pan << "\ntilt=" << cam.tilt
              << "\nfov=" << cam.fov << "\nsun=" << scene.sun.x << "," << scene.sun.y << "," << scene.sun.z << "\nbody=" << h.body.x << ","
              << h.body.y << "," << h.body.z << "," << h.body_r << "\nhead=" << h.head.x << "," << h.head.y << "," << h.head.z << "," << h.head_r
              << "\nsword=" << h.sword_a.x << "," << h.sword_a.y << "," << h.sword_a.z << "," << h.sword_b.x << "," << h.sword_b.y << ","
              << h.sword_b.z << "," << h.sword_r << "\npanels=" << scene.n_panels << "\n";
            for (int p = 0; p < scene.n_panels; ++p) {
                const d3::Panel& P = scene.panels[p];
                s << "panel" << p << "=" << P.c.x << "," << P.c.y << "," << P.c.z << "," << P.n.x << "," << P.n.y << "," << P.n.z << ","
                  << P.u.x << "," << P.u.y << "," << P.u.z << "," << P.v.x << "," << P.v.y << "," << P.v.z << "," << P.hu << "," << P.hv << ","
                  << P.post_base.x << "," << P.post_base.y << "," << P.post_base.z << "\n";
            }
        }

        if (hero.hp == 0) { winner = guardian.name_en; ++f; break; }
        if (guardian.hp == 0) { winner = hero.name_en; ++f; break; }

        int steps = static_cast<int>(std::lround(dt / sim_dt));
        for (int st = 0; st < steps; ++st) {
            pan_m = droid::motor_step(mp, pan_m, pan_ref, sim_dt);
            tilt_m = droid::motor_step(mp, tilt_m, tilt_ref, sim_dt);
            head = droid::mount_step(mount, head, t + st * sim_dt, rock_A, rock_f, sim_dt);
            motor_w_sum += (mp.R * (pan_m.i * pan_m.i + tilt_m.i * tilt_m.i)
                            + mp.Ke * std::fabs(pan_m.omega * pan_m.i + tilt_m.omega * tilt_m.i)) * sim_dt;   // joules
        }
    }
    if (csv.is_open()) { csv.close(); if (!csv) { std::cerr << "error writing " << o.csv << "\n"; return 4; } }

    std::printf("\n=== %s after %.1f s (%d frames) ===\n", winner.empty() ? "time up" : ("winner: " + winner).c_str(), f / fps, f);
    std::printf("HP  %s %d/%d   %s %d/%d\n", guardian.name_en.c_str(), guardian.hp, guardian.max_hp, hero.name_en.c_str(), hero.hp, hero.max_hp);
    std::printf("beam hits %d, parried by foam sword %d, blocked by hills %d; foam slashes landed %d\n", g_hits, parries, hill_blocks, h_hits);
    std::printf("target seen %d/%d frames, line of sight clear %d/%d, mean true aim error %.2f deg\n", found_frames, f, los_frames, f, sum_err / f * 180 / PI);
    std::printf("render %s %.2f ms/frame, vision %s %.2f ms/frame (%dx%d)\n", gpu_render ? "GPU" : "CPU", render_ms / f, gpu_vision ? "GPU" : "CPU",
                vision_ms / f, o.n, o.n);
    double secs = f / fps;
    std::printf("%s: mean %.2f kW (min %.2f, max %.2f), %.1f Wh over the battle\n", solar_ja.c_str(), pv_sum_kw / f, pv_min_kw, pv_max_kw,
                pv_sum_kw / f * 1e3 * secs / 3600);
    std::printf("  unshaded ideal %.2f kW; row-to-row self-shading at this sun angle costs %.1f%%; the hero's shadow dimmed the farm in %d of %d frames\n",
                pv_ideal_w / 1e3, 100 * (1 - pv_base_w / pv_ideal_w), shaded_frames, f);
    std::printf("gimbal motors alone drew %.3f W on average (electronics, camera and GPU not modelled)\n", motor_w_sum / secs);
    return 0;
}
