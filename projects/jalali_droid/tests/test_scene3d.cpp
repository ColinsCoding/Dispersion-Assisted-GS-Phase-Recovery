// test_scene3d.cpp -- the 3D world and the battle rules, each against a closed form or an exact check.
// Exit code = number of failed checks.
#include "game3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using namespace d3;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what, double got = 0, double want = 0) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL  %s  (got %.6g, want %.6g)\n", what, got, want); }
    else std::printf("ok    %s\n", what);
}

int main(int argc, char** argv) {
    std::string defs_path = argc > 1 ? argv[1] : "assets/characters.def";
    Terrain T;

    // terrain: analytic normal is a unit vector and matches a central-difference gradient
    float x = 1.7f, y = -2.3f, e = 1e-3f;
    V3 nrm = terrain_normal(T, x, y);
    float hx = (height(T, x + e, y) - height(T, x - e, y)) / (2 * e), hy = (height(T, x, y + e) - height(T, x, y - e)) / (2 * e);
    V3 fd = normalize(mk(-hx, -hy, 1));
    check(std::fabs(len(nrm) - 1) < 1e-6, "terrain normal is a unit vector", len(nrm), 1);
    check(len(nrm - fd) < 2e-3, "analytic terrain normal = finite-difference normal", len(nrm - fd), 0);

    // sphere: ray from the origin along +x hits a unit sphere at x = 5 when t = 4
    check(std::fabs(hit_sphere(mk(0, 0, 0), mk(1, 0, 0), mk(5, 0, 0), 1.0f) - 4.0f) < 1e-5, "ray-sphere hit at t = 4");
    check(hit_sphere(mk(0, 0, 0), mk(1, 0, 0), mk(5, 3, 0), 1.0f) < 0, "ray misses an offset sphere");

    // capsule: segment along z at x = 5, radius 0.5: side hit at t = 4.5; end-cap hit from above
    check(std::fabs(hit_capsule(mk(0, 0, 0), mk(1, 0, 0), mk(5, 0, -1), mk(5, 0, 1), 0.5f) - 4.5f) < 1e-4, "ray-capsule side hit at t = 4.5");
    check(std::fabs(hit_capsule(mk(5, 0, 5), mk(0, 0, -1), mk(5, 0, -1), mk(5, 0, 1), 0.5f) - 3.5f) < 1e-4, "ray-capsule end-cap hit at t = 3.5");

    // terrain ray: the hit point lies on the surface
    V3 ro = mk(0, 0, 5), rd = normalize(mk(1, 0.3f, -0.4f));
    float t = hit_terrain(T, ro, rd, 80.0f);
    V3 p = ro + t * rd;
    check(t > 0 && std::fabs(p.z - height(T, p.x, p.y)) < 1e-3, "terrain ray hit lies on h(x, y)", t > 0 ? std::fabs(p.z - height(T, p.x, p.y)) : -1, 0);

    // line of sight: clear high above the hills, blocked straight through the ground
    check(!terrain_blocks(T, mk(-5, 0, 6), mk(5, 0, 6)), "line of sight clear 6 m above the field");
    check(terrain_blocks(T, mk(-5, 0, 0.3f), mk(5, 0, -3)), "line of sight blocked through the ground");

    // camera: centre pixel ray ~ forward unit vector; the basis is orthonormal
    Camera c{mk(0, 0, 2), 0.6f, -0.1f, 0.6f, 128};
    V3 f, r, u;
    camera_basis(c, f, r, u);
    V3 mid = normalize(pixel_ray(c, 63, 63) + pixel_ray(c, 64, 64));
    check(len(mid - f) < 1e-4, "centre pixel ray = camera forward", len(mid - f), 0);
    check(std::fabs(dot(f, r)) < 1e-6 && std::fabs(dot(f, u)) < 1e-6 && std::fabs(dot(r, u)) < 1e-6, "camera basis is orthonormal");
    check(std::fabs(len(aim(0.6f, -0.1f)) - 1) < 1e-6, "aim(pan, tilt) is a unit vector");

    // jump: v0 t - g t^2 / 2 peaks at v0^2 / 2g (here sampled finely)
    double v0 = 4.0, g = 9.81, apex = 0;
    for (int k = 0; k <= 10000; ++k) { double tt = k * (2 * v0 / g) / 10000; apex = std::max(apex, v0 * tt - 0.5 * g * tt * tt); }
    check(std::fabs(apex - v0 * v0 / (2 * g)) < 1e-6, "hop apex = v0^2 / 2g = 0.815 m", apex, v0 * v0 / (2 * g));

    // definitions and battle arithmetic
    g3::Defs defs;
    std::string err;
    bool loaded = g3::load_defs(defs_path, defs, err);
    check(loaded, "characters.def loads");
    if (loaded) {
        g3::Fighter gu = g3::fighter_from(defs, "guardian"), he = g3::fighter_from(defs, "hero");
        check(gu.name_ja.size() > 6 && (unsigned char)gu.name_ja[0] >= 0xE0, "Japanese name read as UTF-8 (multi-byte)");
        check(g3::type_multiplier(defs, "Fighting", "Photon/Steel") == 2.0, "Fighting vs Photon/Steel = 2.0 x 1.0");
        check(g3::type_multiplier(defs, "Photon", "Fighting") == 1.0, "Photon vs Fighting = 1.0");
        // hand values: floor(floor(14 * 40 * 70 / 60) / 50) + 2 = 15, x1.5 STAB = 22.5 -> 22 at r = 1
        int d1 = g3::damage(gu, he, 1.0, 1.0), d2 = g3::damage(he, gu, 2.0, 1.0), d3v = g3::damage(he, gu, 2.0, 0.0);
        check(d1 == 22, "Phase Stretch Beam on Souma = 22 (hand-computed)", d1, 22);
        check(d2 == 48, "Foam Slash on the Guardian = 48 (super effective, hand-computed)", d2, 48);
        check(d3v == 40, "minimum roll (x0.85) gives 40", d3v, 40);
    }

    // ---- solarpunk additions
    Panel rp;
    rp.c = mk(5, 0, 0); rp.n = mk(-1, 0, 0); rp.u = mk(0, 1, 0); rp.v = mk(0, 0, 1); rp.hu = 1; rp.hv = 1; rp.post_base = mk(5, 0, -5);
    check(std::fabs(hit_rect(mk(0, 0, 0), mk(1, 0, 0), rp) - 5.0f) < 1e-5, "ray-rectangle hit at t = 5");
    check(hit_rect(mk(0, 1.5f, 0), mk(1, 0, 0), rp) < 0, "ray misses beside the rectangle");

    Scene sp;
    sp.sun = normalize(mk(0.4f, -0.3f, 0.85f));
    sp.hero.body = mk(500, 500, 50); sp.hero.body_r = 0.3f;              // hero far away
    sp.hero.head = mk(500, 500, 51); sp.hero.head_r = 0.2f;
    sp.hero.sword_a = mk(500, 500, 52); sp.hero.sword_b = mk(500, 501, 52); sp.hero.sword_r = 0.05f;
    if (loaded) {
        int np = g3::build_solar_field(defs, sp);
        check(np == 21, "solar field: 3 rows x 7 panels", np, 21);
        double tilt = std::acos(sp.panels[0].n.z) * 180 / 3.14159265358979;
        check(std::fabs(tilt - 38.6) < 1e-3, "panels tilted at the site latitude (38.6 deg)", tilt, 38.6);
        double un;
        double P = g3::panel_power(sp, 0, 900, 110, 0.21, un);
        const Panel& P0 = sp.panels[0];
        double want = 0.21 * 1.7 * 1.0 * (900 * dot(P0.n, sp.sun) + 110 * 0.5 * (1 + P0.n.z));
        check(un == 1.0 && std::fabs(P - want) < 1e-6 * want, "unshaded panel power = eta A (DNI cos + DHI (1+nz)/2)", P, want);
        sp.panels[0].dirt = 1.0f;
        double Pd = g3::panel_power(sp, 0, 900, 110, 0.21, un);
        sp.panels[0].dirt = 0.0f;
        check(std::fabs(Pd - 0.5 * P) < 1e-9 * P, "fully dusty panel delivers half the power (soiling)", Pd, 0.5 * P);
        check(in_shadow(sp, P0.c - 1.2f * sp.sun, mk(0, 0, 1)) && !in_shadow(sp, mk(0, 0, 30), mk(0, 0, 1)),
              "shadow ray: shaded under a panel, sunlit in open sky");
        sp.hero.body = P0.c + 2.0f * sp.sun;                              // hero stands between the sun and panel 0
        double Ps = g3::panel_power(sp, 0, 900, 110, 0.21, un);
        check(un < 1.0 && Ps < P, "the hero's shadow costs panel 0 power", Ps, P);
    }
    float worst_sky = 0;                                                   // sun disk, halo, horizon: never "orange"
    for (V3 sun_dir : {normalize(mk(0.4f, -0.3f, 0.85f)), normalize(mk(0.62f, -0.47f, 0.42f))}) {   // noon and late afternoon
        sp.sun = sun_dir;
        for (int a = 0; a < 72; ++a)
            for (int e = -10; e <= 90; e += 4) {
                float az = a * 0.0872665f, el = e * 0.0174533f;
                worst_sky = std::max(worst_sky, target_feature(sky(sp, mk(cosf(el) * cosf(az), cosf(el) * sinf(az), sinf(el)))));
            }
        worst_sky = std::max(worst_sky, target_feature(sky(sp, sp.sun)));
    }
    check(worst_sky == 0.0f, "no sky direction (sun disk and halo, noon and late sun) looks like the target", worst_sky, 0);
    V3 glint = mk(0.05f, 0.10f, 0.26f) + 0.9f * mk(1.0f, 0.97f, 0.92f);  // panel glass + full specular
    check(target_feature(glint) == 0.0f, "panel glint is not mistaken for the target");

    // CPU renderer: a frame looking at the hero shows the orange tunic at the centre
    Scene s;
    s.sun = normalize(mk(0.4f, -0.3f, 0.85f));
    V3 foot = mk(8, 0, height(T, 8, 0));
    s.hero.body = foot + mk(0, 0, 0.95f); s.hero.body_r = 0.32f;
    s.hero.head = foot + mk(0, 0, 1.55f); s.hero.head_r = 0.18f;
    s.hero.sword_a = s.hero.body + mk(-0.3f, 0.3f, 0); s.hero.sword_b = s.hero.sword_a + mk(0, 0, 1.0f); s.hero.sword_r = 0.05f;
    V3 cp = mk(0, 0, height(T, 0, 0) + 2.0f), to = s.hero.body - cp;
    Camera cam{cp, std::atan2(to.y, to.x), std::asin(to.z / len(to)), 0.25f, 64};
    std::vector<float> cpu;
    g3::render_cpu(s, cam, cpu);
    size_t ctr = (32 * 64 + 32) * 3;
    check(target_feature(mk(cpu[ctr], cpu[ctr + 1], cpu[ctr + 2])) > 0.3f, "centre pixel sees the orange tunic", cpu[ctr], 0.95);
    check(target_feature(mk(cpu[0], cpu[1], cpu[2])) == 0.0f, "corner pixel (grass/sky) has zero target feature");

#ifdef DROID_CUDA
    std::vector<float> gpu;
    bool ok = g3::render_gpu(s, cam, gpu, err);
    double worst = 0;
    int differ = 0;
    if (ok) for (size_t i = 0; i < cpu.size(); ++i) { double dd = std::fabs(cpu[i] - gpu[i]); worst = std::max(worst, dd); differ += dd > 1e-3; }
    check(ok && differ <= static_cast<int>(cpu.size() / 200), "GPU render = CPU render (<0.5% of values differ by >1e-3)", differ, 0);
#else
    std::printf("skip  GPU render check (built without CUDA)\n");
#endif

    std::printf("\n%d / %d checks passed\n", checks - failures, checks);
    return failures;
}
