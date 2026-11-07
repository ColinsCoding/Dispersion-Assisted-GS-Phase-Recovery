// play3d.cpp -- PLAY the solarpunk battle in C++/CUDA: you are 剣士ソウマ (Souma) vs. the Jalali Guardian.
//
// Two phases, two cameras:
//   1st person  農場 (farming): walk the solar farm with mouse look; tend dusty panels (E) -- dust really costs
//               watts -- and the farm's harvest fills your solar charge.
//   3rd person  クランチタイム (crunch time): when anyone drops to 30% HP the camera cuts to third person and
//               最後の賭け (the Last Gamble) unlocks: Q with full charge = a 50/50 throw. Win: the droid is
//               stunned for 3 s and your next slash does double damage. Lose: 10 HP of recoil.
// The Guardian hunts you the whole time with Phase Stretch Transform vision on its own ray-traced camera.
//
// Input   mouse = look (captured); left click = Foam Slash (in reach); right click (hold) = raise the foam sword
//         W A S D = move / strafe, Space = jump, E = tend a panel, Q = Last Gamble, V = switch camera, Esc = quit
// Output  ray-traced view (GPU or CPU), HUD (HP, solar charge, farm kW, prompts in Japanese), synthesized sound
//
//   droid_play [--defs PATH] [--cpu] [--mute] [--autoplay FRAMES] [--screenshot PREFIX]
// --autoplay drives keyboard + mouse from a script (testing) and saves PREFIX_1st.bmp, PREFIX_3rd.bmp, PREFIX_end.bmp.
// Exit codes: 0 ok, 2 bad args / not Windows, 4 cannot write, 5 bad definitions.
#ifndef _WIN32
#include <cstdio>
int main() { std::puts("droid_play needs Windows (Win32 window, mouse capture, sound). Use droid3d on Linux."); return 2; }
#else
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "droid.hpp"
#include "game3d.hpp"

using d3::V3;
using droid::PI;

namespace {

bool g_running = true, g_focus = true;
const int VIEW = 256, SCALE = 3, WIN = VIEW * SCALE;

LRESULT CALLBACK wnd_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE || m == WM_DESTROY) { g_running = false; return 0; }
    if (m == WM_ACTIVATE) g_focus = LOWORD(w) != WA_INACTIVE;
    return DefWindowProcW(h, m, w, l);
}

std::wstring wide(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

// ------------------------------------------------------------------ sound: tiny synthesized WAVs, played async
struct Sound { std::vector<char> wav; };

Sound synth(double dur, const std::function<double(double)>& f) {
    const int sr = 22050, n = static_cast<int>(dur * sr);
    Sound s;
    s.wav.resize(44 + 2 * n);
    char* p = s.wav.data();
    auto u32 = [&](int off, unsigned v) { std::memcpy(p + off, &v, 4); };
    auto u16 = [&](int off, unsigned short v) { std::memcpy(p + off, &v, 2); };
    std::memcpy(p, "RIFF", 4); u32(4, 36 + 2 * n); std::memcpy(p + 8, "WAVEfmt ", 8);
    u32(16, 16); u16(20, 1); u16(22, 1); u32(24, sr); u32(28, 2 * sr); u16(32, 2); u16(34, 16);
    std::memcpy(p + 36, "data", 4); u32(40, 2 * n);
    for (int i = 0; i < n; ++i) {
        double t = i / static_cast<double>(sr), env = std::min(1.0, 40 * t) * std::max(0.0, 1.0 - t / dur);
        short v = static_cast<short>(std::lround(12000 * env * std::max(-1.0, std::min(1.0, f(t)))));
        std::memcpy(p + 44 + 2 * i, &v, 2);
    }
    return s;
}

bool g_mute = false;
void play(const Sound& s) { if (!g_mute) PlaySoundW(reinterpret_cast<LPCWSTR>(s.wav.data()), nullptr, SND_MEMORY | SND_ASYNC | SND_NODEFAULT); }

// ------------------------------------------------------------------ input
struct Input {
    bool fwd = false, back = false, left = false, right = false, jump = false, guard = false, slash = false;
    bool tend = false, gamble = false, toggle = false, quit = false;
    double mdx = 0, mdy = 0;
};

Input read_input(HWND hwnd) {
    Input in;
    if (!g_focus || GetForegroundWindow() != hwnd) return in;
    auto down = [](int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
    in.fwd = down('W'); in.back = down('S'); in.left = down('A'); in.right = down('D');
    in.jump = down(VK_SPACE); in.tend = down('E'); in.gamble = down('Q'); in.toggle = down('V'); in.quit = down(VK_ESCAPE);
    in.slash = down(VK_LBUTTON); in.guard = down(VK_RBUTTON);
    RECT rc;
    GetClientRect(hwnd, &rc);
    POINT c{(rc.right - rc.left) / 2, (rc.bottom - rc.top) / 2};
    ClientToScreen(hwnd, &c);
    POINT m;
    GetCursorPos(&m);
    in.mdx = m.x - c.x; in.mdy = m.y - c.y;                     // mouse look: offset from the centre, then recentre
    SetCursorPos(c.x, c.y);
    return in;
}

// autoplay bot: turn the "mouse" toward a target (the dustiest panel while farming, then the droid), walk to it,
// tend while farming; in the fight take hits until crunch time, then gamble, guard and slash.
Input scripted(bool farming, double hx, double hy, double yaw, double tx, double ty, bool under_fire, double hp_frac, double mouse_k) {
    Input in;
    double err = droid::wrap_pi(std::atan2(ty - hy, tx - hx) - yaw);
    in.mdx = -0.35 * err / mouse_k;
    bool desperate = hp_frac <= 0.3;                           // fight: soak beams at 6 m until crunch time, then close in
    in.fwd = std::hypot(tx - hx, ty - hy) > (farming ? 0.4 : (desperate ? 1.3 : 6.0)) && std::fabs(err) < 1.0;
    in.tend = farming;
    in.guard = desperate && under_fire;
    in.slash = !farming && desperate && !under_fire;
    in.gamble = !farming;
    return in;
}

// ------------------------------------------------------------------ helpers
d3::Hero hero_geometry(V3 foot, float yaw, bool guard, float swing) {
    V3 fwd = d3::mk(std::cos(yaw), std::sin(yaw), 0), right = d3::normalize(d3::cross(fwd, d3::mk(0, 0, 1)));
    d3::Hero h;
    h.body = foot + d3::mk(0, 0, 0.95f); h.body_r = 0.32f;
    h.head = foot + d3::mk(0, 0, 1.55f); h.head_r = 0.18f;
    V3 hand = h.body + 0.30f * right + 0.30f * fwd - d3::mk(0, 0, 0.05f);
    V3 dir = guard ? d3::normalize(-1.0f * right + 0.15f * d3::mk(0, 0, 1))
                   : d3::normalize(0.6f * fwd + std::sin(swing) * d3::mk(0, 0, 1) - 0.3f * d3::mk(0, 0, 1));
    h.sword_a = hand; h.sword_b = hand + 1.0f * dir; h.sword_r = 0.05f;
    return h;
}

void put_bar(std::vector<unsigned>& px, int x0, int y0, int w, int h, double frac, unsigned col) {
    frac = std::max(0.0, std::min(1.0, frac));
    for (int y = y0; y < y0 + h; ++y)
        for (int x = x0; x < x0 + w; ++x) px[static_cast<size_t>(y) * VIEW + x] = (x - x0 < frac * w) ? col : 0x00303030u;
}

void crosshair(std::vector<unsigned>& px, unsigned col) {
    for (int k = -5; k <= 5; ++k)
        if (std::abs(k) > 1) { px[static_cast<size_t>(VIEW / 2) * VIEW + VIEW / 2 + k] = col; px[static_cast<size_t>(VIEW / 2 + k) * VIEW + VIEW / 2] = col; }
}

bool save_bmp(const std::string& path, const std::vector<unsigned>& px) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    BITMAPFILEHEADER fh{};
    BITMAPINFOHEADER ih{};
    ih.biSize = sizeof ih; ih.biWidth = VIEW; ih.biHeight = -VIEW; ih.biPlanes = 1; ih.biBitCount = 32; ih.biCompression = BI_RGB;
    fh.bfType = 0x4D42; fh.bfOffBits = sizeof fh + sizeof ih; fh.bfSize = fh.bfOffBits + VIEW * VIEW * 4;
    f.write(reinterpret_cast<const char*>(&fh), sizeof fh);
    f.write(reinterpret_cast<const char*>(&ih), sizeof ih);
    f.write(reinterpret_cast<const char*>(px.data()), VIEW * VIEW * 4);
    return static_cast<bool>(f);
}

struct Lcg {
    unsigned s;
    explicit Lcg(unsigned seed) : s(seed) {}
    double uniform() { s = 1664525u * s + 1013904223u; return (s >> 8) * (1.0 / 16777216.0); }
};

}  // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(65001);
    std::string defs_path = "assets/characters.def", shot;
    bool force_cpu = false;
    int autoplay = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--defs" && i + 1 < argc) defs_path = argv[++i];
        else if (a == "--cpu") force_cpu = true;
        else if (a == "--mute") g_mute = true;
        else if (a == "--autoplay" && i + 1 < argc) autoplay = std::atoi(argv[++i]);
        else if (a == "--screenshot" && i + 1 < argc) shot = argv[++i];
        else { std::puts("usage: droid_play [--defs PATH] [--cpu] [--mute] [--autoplay FRAMES] [--screenshot PREFIX]"); return 2; }
    }

    // ---------------------------------------------------------------- world and characters
    g3::Defs defs;
    std::string err;
    g3::Fighter guardian, hero;
    d3::Scene scene;
    double hold_s, cooldown_s, laser_range, reach, v0, g, dni, dhi, eta;
    try {
        if (!g3::load_defs(defs_path, defs, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 5; }
        guardian = g3::fighter_from(defs, "guardian");
        hero = g3::fighter_from(defs, "hero");
        hold_s = g3::num(defs, "guardian", "hit_hold_s");
        cooldown_s = g3::num(defs, "guardian", "cooldown_s");
        laser_range = g3::num(defs, "guardian", "laser_range_m");     // play mode: the farm edge is out of reach
        reach = g3::num(defs, "hero", "reach_m");
        v0 = g3::num(defs, "hero", "jump_speed_mps");
        g = g3::num(defs, "world", "gravity");
        scene.sun = d3::normalize(d3::mk(static_cast<float>(g3::num(defs, "world", "sun_x")), static_cast<float>(g3::num(defs, "world", "sun_y")),
                                         static_cast<float>(g3::num(defs, "world", "sun_z"))));
        g3::build_solar_field(defs, scene);
        dni = g3::num(defs, "solar", "dni_w_m2"); dhi = g3::num(defs, "solar", "dhi_w_m2"); eta = g3::num(defs, "solar", "module_efficiency");
    } catch (const std::exception& e) { std::fprintf(stderr, "definitions: %s\n", e.what()); return 5; }
    Lcg rng(11);
    for (int p = 0; p < scene.n_panels; ++p) scene.panels[p].dirt = static_cast<float>(0.25 + 0.5 * rng.uniform());   // a dusty farm to tend
    double ideal_kw = 0;
    {
        d3::Scene clean = scene;
        clean.hero = hero_geometry(d3::mk(500, 500, 0), 0, false, 0);
        double un;
        for (int p = 0; p < clean.n_panels; ++p) { clean.panels[p].dirt = 0; ideal_kw += g3::panel_power(clean, p, dni, dhi, eta, un) / 1e3; }
    }

    // ---------------------------------------------------------------- sounds
    Sound s_laser = synth(0.18, [](double t) { return std::sin(2 * PI * (1800 * t - 3000 * t * t)); });
    Sound s_slash = synth(0.14, [](double t) { return std::sin(2 * PI * 300 * t) * std::sin(2 * PI * 37 * t * 40) * 0.8 + 0.3 * std::sin(1e5 * t * t); });
    Sound s_parry = synth(0.12, [](double t) { return std::sin(2 * PI * 2400 * t) * 0.8 + 0.3 * std::sin(2 * PI * 3600 * t); });
    Sound s_hit = synth(0.25, [](double t) { return std::sin(2 * PI * (140 - 200 * t) * t); });
    Sound s_win = synth(0.6, [](double t) { double f = t < 0.2 ? 523 : (t < 0.4 ? 659 : 784); return std::sin(2 * PI * f * t); });
    Sound s_lose = synth(0.5, [](double t) { return std::sin(2 * PI * (300 - 300 * t) * t); });
    Sound s_tend = synth(0.08, [](double t) { return 0.5 * std::sin(2 * PI * 900 * t); });

    // ---------------------------------------------------------------- window, mouse capture
    HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc; wc.hInstance = inst; wc.lpszClassName = L"JalaliDroidPlay2"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT r{0, 0, WIN, WIN};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, FALSE);
    std::wstring title = wide(hero.name_ja + " vs " + guardian.name_ja + "  —  mouse look, L-click slash, R-click guard, WASD, Space, E tend, Q gamble, V camera, Esc quit");
    HWND hwnd = CreateWindowW(wc.lpszClassName, title.c_str(), (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME) | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                              r.right - r.left, r.bottom - r.top, nullptr, nullptr, inst, nullptr);
    if (!hwnd) { std::fputs("could not open a window\n", stderr); return 2; }
    SetForegroundWindow(hwnd);
    if (!autoplay) ShowCursor(FALSE);
    HDC dc = GetDC(hwnd);
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER); bi.bmiHeader.biWidth = VIEW; bi.bmiHeader.biHeight = -VIEW;
    bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    HFONT font = CreateFontW(22, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, ANTIALIASED_QUALITY, 0, L"Yu Gothic UI");
    SelectObject(dc, font);
    SetBkMode(dc, TRANSPARENT);

    // ---------------------------------------------------------------- state
    const double dt = 1.0 / 30.0, sim_dt = 1e-4, mouse_k = 0.0035;
    const V3 droid_base = d3::mk(0, 0, d3::height(scene.T, 0, 0));
    const V3 cam_pos = droid_base + d3::mk(0, 0, 2.0f);
    float hx = -6.0f, hy = 6.5f, yaw = -0.7f, pitch = -0.05f, jump = 0.0f, vz = 0.0f;   // start beside the solar farm
    bool airborne = false, first_person = true, crunch = false, toggle_was = false, gamble_was = false, firing_was = false;
    double next_slash = 0, next_g = 0, hold = 0, fov = 45 * PI / 180, stun_until = -1, charge = 0, tend_snd = 0;
    double last_pan = 0, last_tilt = 0, rate_pan = 0, rate_tilt = 0;
    bool have_last = false, overdrive = false;
    droid::MotorParams mp;
    droid::MotorState pan_m, tilt_m;
    tilt_m.theta = -0.12;
    pan_m.theta = std::atan2(hy, hx);
    std::vector<float> rgb, vis;
    std::vector<unsigned> px(static_cast<size_t>(VIEW) * VIEW);
    bool gpu = !force_cpu;
#ifndef DROID_CUDA
    gpu = false;
#endif
    std::wstring banner;
    double banner_until = 0;
    std::string outcome;
    int frame = 0, parries = 0, beam_hits = 0, slashes = 0, gambles_won = 0, gambles_lost = 0;
    double pv_kw = 0, harvested_wh = 0;
    bool saved_1st = false, saved_3rd = false;
    auto clock0 = std::chrono::steady_clock::now();

    while (g_running) {
        auto ft0 = std::chrono::steady_clock::now();
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = frame * dt;
        Input in;
        if (autoplay) {
            bool farming = frame < autoplay * 2 / 3;
            double tx = 0, ty = 0, worst = -1;
            if (farming)
                for (int p = 0; p < scene.n_panels; ++p)
                    if (scene.panels[p].dirt > worst + 0.05) { worst = scene.panels[p].dirt; tx = scene.panels[p].c.x; ty = scene.panels[p].c.y; }
            in = scripted(farming, hx, hy, yaw, tx, ty, firing_was, static_cast<double>(hero.hp) / hero.max_hp, mouse_k);
            in.gamble = in.gamble && frame % 2 == 0;                     // tap Q (the gamble fires on a fresh press)
        } else in = read_input(hwnd);
        if (in.quit || (autoplay && frame >= autoplay)) break;
        if (in.toggle && !toggle_was) first_person = !first_person;
        toggle_was = in.toggle;

        // ---------------------------------------------------------- you: mouse look, WASD, jump
        if (outcome.empty()) {
            yaw -= static_cast<float>(in.mdx * mouse_k);
            pitch = std::max(-1.2f, std::min(1.2f, pitch - static_cast<float>(in.mdy * mouse_k)));
            V3 fwd = d3::mk(std::cos(yaw), std::sin(yaw), 0), rgt = d3::mk(std::sin(yaw), -std::cos(yaw), 0);
            float mf = static_cast<float>((in.fwd ? 1 : 0) - (in.back ? 1 : 0)), ms = static_cast<float>((in.right ? 1 : 0) - (in.left ? 1 : 0));
            float spd = static_cast<float>(4.5 * dt);
            hx += spd * (mf * fwd.x + ms * rgt.x); hy += spd * (mf * fwd.y + ms * rgt.y);
            float rad = std::hypot(hx, hy);
            if (rad > 18.5f) { hx *= 18.5f / rad; hy *= 18.5f / rad; }      // arena edge, just past the outer panels
            if (rad < 1.0f) { hx /= std::max(rad, 1e-3f); hy /= std::max(rad, 1e-3f); }
            if (in.jump && !airborne) { airborne = true; vz = static_cast<float>(v0); }
            if (airborne) { vz -= static_cast<float>(g * dt); jump += vz * static_cast<float>(dt); if (jump <= 0) { jump = 0; airborne = false; } }
        }
        V3 foot = d3::mk(hx, hy, d3::height(scene.T, hx, hy) + jump);
        scene.hero = hero_geometry(foot, yaw, in.guard, static_cast<float>(3.0 * t));

        // ---------------------------------------------------------- farming: dust settles, you tend, the farm charges you
        int near_panel = -1;
        double near_d = 2.2;
        for (int p = 0; p < scene.n_panels; ++p) {
            d3::Panel& P = scene.panels[p];
            P.dirt = std::min(1.0f, P.dirt + static_cast<float>(0.004 * dt));        // slow soiling
            double dd = std::hypot(P.c.x - hx, P.c.y - hy);
            if (dd < near_d) { near_d = dd; near_panel = p; }
        }
        if (in.tend && near_panel >= 0 && scene.panels[near_panel].dirt > 0) {
            scene.panels[near_panel].dirt = std::max(0.0f, scene.panels[near_panel].dirt - static_cast<float>(0.8 * dt));
            if (t > tend_snd) { play(s_tend); tend_snd = t + 0.25; }
        }
        if (frame % 10 == 0) {
            double w = 0, un;
            for (int p = 0; p < scene.n_panels; ++p) w += g3::panel_power(scene, p, dni, dhi, eta, un);
            pv_kw = w / 1e3;
        }
        harvested_wh += pv_kw * 1e3 * dt / 3600;
        charge = std::min(100.0, charge + (pv_kw / ideal_kw) * (100.0 / 45.0) * dt);   // a clean farm fills it in ~45 s

        // ---------------------------------------------------------- crunch time
        bool low = hero.hp <= 0.3 * hero.max_hp || guardian.hp <= 0.3 * guardian.max_hp;
        if (low && !crunch && outcome.empty()) {
            crunch = true; first_person = false;
            banner = L"クランチタイム!  CRUNCH TIME — 3人称"; banner_until = t + 2.0;
        }

        // ---------------------------------------------------------- the droid's own camera -> PST -> tracking
        scene.droid.visible = 0; scene.droid.laser_on = 0;
        d3::Camera dcam{cam_pos, static_cast<float>(pan_m.theta), static_cast<float>(tilt_m.theta), static_cast<float>(fov), 128};
        bool ok = false;
#ifdef DROID_CUDA
        if (gpu) { ok = g3::render_gpu(scene, dcam, vis, err); if (!ok) gpu = false; }
#endif
        if (!ok) g3::render_cpu(scene, dcam, vis);
        droid::Image feat;
        feat.n = 128;
        feat.px.resize(128 * 128);
        for (size_t p = 0; p < feat.px.size(); ++p) feat.px[p] = 0.2f + 0.8f * d3::target_feature(d3::mk(vis[3 * p], vis[3 * p + 1], vis[3 * p + 2]));
        std::vector<float> edge;
        bool vok = false;
#ifdef DROID_CUDA
        if (gpu) vok = droid::pst_gpu(feat, 0.4, 20.0, 0.05, edge, err);
#endif
        if (!vok) edge = droid::pst_cpu(feat, 0.4, 20.0, 0.05);
        bool stunned = t < stun_until;
        double cx, cy, pan_ref = pan_m.theta + (stunned ? 0.0 : 0.05), tilt_ref = stunned ? tilt_m.theta : -0.10, meas_err = 1e9;
        V3 fw, rt, up;
        d3::camera_basis(dcam, fw, rt, up);
        bool found = !stunned && droid::locate_target(edge, 128, 0.3, cx, cy);
        if (found) {
            float s = std::tan(0.5f * static_cast<float>(fov));
            V3 dd = d3::normalize(fw + static_cast<float>((2 * (cx + 0.5) / 128 - 1) * s) * rt + static_cast<float>((1 - 2 * (cy + 0.5) / 128) * s) * up);
            double mpn, mtl;
            droid::pan_tilt_of({dd.x, dd.y, dd.z}, mpn, mtl);
            meas_err = std::acos(std::min(1.0f, d3::dot(dd, fw)));
            if (have_last) { rate_pan = 0.6 * rate_pan + 0.4 * droid::wrap_pi(mpn - last_pan) / dt; rate_tilt = 0.6 * rate_tilt + 0.4 * (mtl - last_tilt) / dt; }
            last_pan = mpn; last_tilt = mtl; have_last = true;
            pan_ref = mpn + rate_pan * dt; tilt_ref = mtl + rate_tilt * dt;
            fov = meas_err < PI / 180 ? 14 * PI / 180 : 45 * PI / 180;
        } else { have_last = false; rate_pan = rate_tilt = 0; fov = 45 * PI / 180; }

        // ---------------------------------------------------------- laser, slash, gamble
        bool in_range = std::hypot(hx - droid_base.x, hy - droid_base.y) < laser_range;
        bool firing = outcome.empty() && found && in_range && meas_err < 0.4 * PI / 180;
        if (firing && !firing_was) play(s_laser);
        firing_was = firing;
        V3 aimv = d3::aim(static_cast<float>(pan_m.theta), static_cast<float>(tilt_m.theta));
        d3::Hit lh = d3::trace(scene, cam_pos, aimv, 80.0f);
        if (firing) {
            if (lh.id == d3::BODY || lh.id == d3::HEAD) {
                hold += dt;
                if (hold >= hold_s && t >= next_g) {
                    int dmg = g3::damage(guardian, hero, g3::type_multiplier(defs, guardian.move_type, hero.type), rng.uniform());
                    hero.hp = std::max(0, hero.hp - dmg); ++beam_hits; next_g = t + cooldown_s; hold = 0;
                    banner = wide(guardian.move_ja + "!  -" + std::to_string(dmg)); banner_until = t + 1.2;
                    play(s_hit);
                }
            } else {
                hold = 0;
                if (lh.id == d3::SWORD && t >= next_g) { ++parries; next_g = t + 0.5; banner = L"受け流し!  PARRY"; banner_until = t + 0.6; play(s_parry); }
            }
        } else hold = 0;
        double dist = std::hypot(hx - droid_base.x, hy - droid_base.y);
        if (outcome.empty() && in.slash && dist < reach && t >= next_slash) {
            double m = g3::type_multiplier(defs, hero.move_type, guardian.type) * (overdrive ? 2.0 : 1.0);
            int dmg = g3::damage(hero, guardian, m, rng.uniform());
            guardian.hp = std::max(0, guardian.hp - dmg); ++slashes; next_slash = t + 1.2;
            banner = wide(hero.move_ja + (overdrive ? "  ×2 オーバードライブ!" : "!") + "  -" + std::to_string(dmg)); banner_until = t + 1.2;
            overdrive = false;
            play(s_slash);
        }
        bool can_gamble = crunch && charge >= 100.0 && hero.hp <= 0.3 * hero.max_hp && outcome.empty();
        if (in.gamble && !gamble_was && can_gamble) {
            charge = 0;
            if (rng.uniform() < 0.5) {
                ++gambles_won; stun_until = t + 3.0; overdrive = true;
                banner = L"最後の賭け: 成功!  ガーディアン停止 3秒 + 次の斬撃 ×2"; play(s_win);
            } else {
                ++gambles_lost; hero.hp = std::max(0, hero.hp - 10);
                banner = L"最後の賭け: 失敗…  反動 -10 HP"; play(s_lose);
            }
            banner_until = t + 2.0;
        }
        gamble_was = in.gamble;
        if (outcome.empty() && hero.hp == 0) { outcome = guardian.name_ja + " の勝ち"; banner_until = 1e9; banner = wide("YOU LOSE — " + outcome); play(s_lose); }
        if (outcome.empty() && guardian.hp == 0) { outcome = hero.name_ja + " の勝ち"; banner_until = 1e9; banner = wide("YOU WIN — " + outcome); play(s_win); }

        // ---------------------------------------------------------- gimbal motor ODEs (RK4, 0.1 ms steps)
        for (int st = 0; st < static_cast<int>(std::lround(dt / sim_dt)); ++st) {
            pan_m = droid::motor_step(mp, pan_m, pan_ref, sim_dt);
            tilt_m = droid::motor_step(mp, tilt_m, tilt_ref, sim_dt);
        }

        // ---------------------------------------------------------- your view: 1st person (farming) or 3rd person (crunch)
        scene.droid.visible = 1;
        scene.droid.base = droid_base;
        scene.droid.head = cam_pos;
        scene.droid.laser_on = firing ? 1 : 0;
        scene.droid.laser_a = cam_pos + 0.33f * aimv;
        scene.droid.laser_b = cam_pos + std::min(lh.t, 60.0f) * aimv;
        V3 look = d3::mk(std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch));
        d3::Camera pc;
        if (first_person) {
            pc = d3::Camera{foot + d3::mk(0, 0, 1.62f), yaw, pitch, static_cast<float>(75 * PI / 180), VIEW};
        } else {
            V3 pcam = foot + d3::mk(0, 0, 1.4f) - 4.2f * look + d3::mk(0, 0, 0.9f);
            float cg = d3::height(scene.T, pcam.x, pcam.y) + 0.4f;
            if (pcam.z < cg) pcam.z = cg;
            V3 l2 = d3::normalize((foot + d3::mk(0, 0, 1.3f)) - pcam);
            pc = d3::Camera{pcam, std::atan2(l2.y, l2.x), std::asin(l2.z), static_cast<float>(60 * PI / 180), VIEW};
        }
        ok = false;
#ifdef DROID_CUDA
        if (gpu) ok = g3::render_gpu(scene, pc, rgb, err);
#endif
        if (!ok) g3::render_cpu(scene, pc, rgb);
        for (size_t p = 0; p < px.size(); ++p) {
            auto c8 = [&](float v) { return static_cast<unsigned>(std::lround(255.0 * std::min(1.0f, std::max(0.0f, v)))); };
            px[p] = (c8(rgb[3 * p]) << 16) | (c8(rgb[3 * p + 1]) << 8) | c8(rgb[3 * p + 2]);
        }
        put_bar(px, 6, 6, 110, 6, static_cast<double>(guardian.hp) / guardian.max_hp, 0x0050C8FFu);
        put_bar(px, VIEW - 116, 6, 110, 6, static_cast<double>(hero.hp) / hero.max_hp, 0x00FF9A20u);
        put_bar(px, VIEW - 116, 14, 110, 4, charge / 100.0, charge >= 100 ? 0x00FFFF40u : 0x00C8B030u);
        if (first_person) crosshair(px, firing ? 0x0060FF80u : 0x00FFFFFFu);
        StretchDIBits(dc, 0, 0, WIN, WIN, 0, 0, VIEW, VIEW, px.data(), &bi, DIB_RGB_COLORS, SRCCOPY);
        auto text = [&](int x, int y, const std::wstring& s, COLORREF col) {
            SetTextColor(dc, RGB(0, 0, 0)); TextOutW(dc, x + 2, y + 2, s.c_str(), static_cast<int>(s.size()));
            SetTextColor(dc, col); TextOutW(dc, x, y, s.c_str(), static_cast<int>(s.size()));
        };
        text(18, 42, wide(guardian.name_ja + "  HP " + std::to_string(guardian.hp) + "/" + std::to_string(guardian.max_hp) + (stunned ? "  (停止中)" : "")), RGB(150, 210, 255));
        text(WIN - 330, 58, wide(hero.name_ja + "  HP " + std::to_string(hero.hp) + "/" + std::to_string(hero.max_hp)), RGB(255, 190, 90));
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s   太陽光 %.2f / %.2f kW   充電 %.0f%%", first_person ? "1人称・農場" : (crunch ? "3人称・クランチタイム" : "3人称"),
                      pv_kw, ideal_kw, charge);
        text(18, WIN - 40, wide(buf), RGB(230, 255, 210));
        if (near_panel >= 0) {
            std::snprintf(buf, sizeof buf, "E: パネル清掃  汚れ %.0f%%", 100.0 * scene.panels[near_panel].dirt);
            text(18, WIN - 72, wide(buf), RGB(255, 240, 160));
        }
        if (dist < reach) text(WIN / 2 - 80, WIN - 72, L"左クリック: 斬れる!", RGB(255, 200, 120));
        else if (!in_range) text(WIN - 230, WIN - 72, L"射程外 (安全地帯)", RGB(170, 255, 170));
        if (can_gamble) text(WIN / 2 - 150, 100, L"Q: 最後の賭け  (50/50)", RGB(255, 255, 90));
        if (firing) text(WIN / 2 - 60, 76, L"⚠ LASER ⚠", RGB(120, 255, 140));
        if (t < banner_until) text(WIN / 2 - 220, WIN / 2 - 120, banner, RGB(255, 255, 255));

        if (!shot.empty() && autoplay && frame > 150) {                   // one picture of each camera, named by what it shows
            bool& saved = first_person ? saved_1st : saved_3rd;
            if (!saved) { save_bmp(shot + (first_person ? "_1st.bmp" : "_3rd.bmp"), px); saved = true; }
        }
        ++frame;
        if (!autoplay) {
            auto spent = std::chrono::steady_clock::now() - ft0;
            auto budget = std::chrono::milliseconds(33);
            if (spent < budget) std::this_thread::sleep_for(budget - spent);
        }
    }
    ShowCursor(TRUE);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock0).count();
    if (!shot.empty() && !save_bmp(shot + "_end.bmp", px)) { std::fprintf(stderr, "cannot write %s\n", shot.c_str()); return 4; }
    double dirt_avg = 0;
    for (int p = 0; p < scene.n_panels; ++p) dirt_avg += scene.panels[p].dirt / scene.n_panels;
    std::printf("%s | %s HP %d/%d, %s HP %d/%d | beams %d, parries %d, slashes %d | gambles won %d lost %d | farm %.2f kW now (ideal %.2f), "
                "harvested %.1f Wh, average dust %.0f%% | %d frames in %.1f s (%.0f fps, %s)\n",
                outcome.empty() ? "quit" : outcome.c_str(), guardian.name_en.c_str(), guardian.hp, guardian.max_hp, hero.name_en.c_str(), hero.hp,
                hero.max_hp, beam_hits, parries, slashes, gambles_won, gambles_lost, pv_kw, ideal_kw, harvested_wh, 100 * dirt_avg, frame, secs,
                frame / secs, gpu ? "GPU" : "CPU");
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return 0;
}
#endif
