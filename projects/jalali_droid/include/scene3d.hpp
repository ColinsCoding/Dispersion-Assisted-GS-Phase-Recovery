// scene3d.hpp -- the solarpunk 3D world, written once and compiled for both CPU and GPU (HD = __host__ __device__).
//
//   terrain   rolling hills z = h(x, y), a sum of sinusoids, with an analytic normal
//   hero      body sphere + head sphere + foam-sword capsule
//   solar     a field of tilted photovoltaic panels (rectangles) on posts (capsules)
//   camera    pinhole PTZ camera: forward = aim unit vector, right = forward x z, up = right x forward
//   tracing   analytic sphere / capsule / rectangle hits; ray-marched + bisected terrain
//   lighting  warm sun with SHADOW RAYS, sky and ground-bounce ambient, Blinn-Phong glint on panel glass,
//             physically-flavoured sky (zenith -> horizon gradient, sun disk and halo), distance haze
// All in float so the CUDA kernel and the CPU loop compute the same numbers.
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace d3 {

struct V3 { float x, y, z; };
HD inline V3 mk(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
HD inline V3 operator+(V3 a, V3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
HD inline V3 operator-(V3 a, V3 b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
HD inline V3 operator*(float s, V3 a) { return mk(s * a.x, s * a.y, s * a.z); }
HD inline V3 mul(V3 a, V3 b) { return mk(a.x * b.x, a.y * b.y, a.z * b.z); }
HD inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD inline V3 cross(V3 a, V3 b) { return mk(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
HD inline float len(V3 a) { return sqrtf(dot(a, a)); }
HD inline V3 normalize(V3 a) { return (1.0f / len(a)) * a; }
HD inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
HD inline V3 mix(V3 a, V3 b, float k) { return a + k * (b - a); }

// ------------------------------------------------------------------ terrain
struct Terrain { float a1 = 0.9f, k1 = 0.35f, a2 = 0.5f, k2 = 0.21f, a3 = 0.35f, kx3 = 0.55f, ky3 = 0.3f; };

HD inline float height(const Terrain& T, float x, float y) {
    return T.a1 * sinf(T.k1 * x) * cosf(0.7f * T.k1 * y) + T.a2 * sinf(T.k2 * (x + y)) + T.a3 * cosf(T.kx3 * x - T.ky3 * y);
}

HD inline V3 terrain_normal(const Terrain& T, float x, float y) {
    float hx = T.a1 * T.k1 * cosf(T.k1 * x) * cosf(0.7f * T.k1 * y) + T.a2 * T.k2 * cosf(T.k2 * (x + y)) - T.a3 * T.kx3 * sinf(T.kx3 * x - T.ky3 * y);
    float hy = -T.a1 * 0.7f * T.k1 * sinf(T.k1 * x) * sinf(0.7f * T.k1 * y) + T.a2 * T.k2 * cosf(T.k2 * (x + y)) + T.a3 * T.ky3 * sinf(T.kx3 * x - T.ky3 * y);
    return normalize(mk(-hx, -hy, 1.0f));
}

// ------------------------------------------------------------------ scene state
constexpr int MAX_PANELS = 32;
struct Panel { V3 c, n, u, v; float hu, hv; V3 post_base; float dirt = 0.0f; };  // centre, normal, axes, half sizes; dust 0..1
struct Hero { V3 body; float body_r; V3 head; float head_r; V3 sword_a, sword_b; float sword_r; };
// the droid itself, drawn only in third-person views (its own camera never sees it); laser = emissive beam
struct Droid { int visible = 0; V3 base, head; float head_r = 0.32f; int laser_on = 0; V3 laser_a, laser_b; };
struct Scene { Terrain T; V3 sun; Hero hero; int n_panels = 0; Panel panels[MAX_PANELS]; Droid droid; };
struct Camera { V3 pos; float pan, tilt, fov; int n; };

enum HitId { SKY = 0, GROUND = 1, BODY = 2, HEAD = 3, SWORD = 4, PANEL = 5, POST = 6, DROID = 7, LASER = 8 };
struct Hit { float t; int id; V3 n; int k; };                     // k = panel index for PANEL / POST

HD inline V3 aim(float pan, float tilt) { return mk(cosf(tilt) * cosf(pan), cosf(tilt) * sinf(pan), sinf(tilt)); }

HD inline void camera_basis(const Camera& c, V3& f, V3& r, V3& u) {
    f = aim(c.pan, c.tilt);
    r = normalize(cross(f, mk(0, 0, 1)));
    u = cross(r, f);
}

HD inline V3 pixel_ray(const Camera& c, int i, int j) {          // i = column (x), j = row (y, down)
    V3 f, r, u;
    camera_basis(c, f, r, u);
    float s = tanf(0.5f * c.fov);
    float px = (2.0f * (i + 0.5f) / c.n - 1.0f) * s, py = (1.0f - 2.0f * (j + 0.5f) / c.n) * s;
    return normalize(f + px * r + py * u);
}

// ------------------------------------------------------------------ intersections (t < 0 = miss)
HD inline float hit_sphere(V3 ro, V3 rd, V3 c, float r) {
    V3 oc = ro - c;
    float b = dot(oc, rd), q = dot(oc, oc) - r * r, h = b * b - q;
    if (h < 0) return -1.0f;
    float t = -b - sqrtf(h);
    return t > 1e-4f ? t : -1.0f;
}

// capsule = cylinder body + two end spheres; nearest positive hit of the three (cylinder: I. Quilez closed form)
HD inline float hit_capsule(V3 ro, V3 rd, V3 pa, V3 pb, float r) {
    V3 ba = pb - pa, oa = ro - pa;
    float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
    float a = baba - bard * bard, b = baba * rdoa - baoa * bard, c = baba * oaoa - baoa * baoa - r * r * baba;
    float best = -1.0f;
    float h = b * b - a * c;
    if (h >= 0.0f && a > 1e-9f * baba) {
        float t = (-b - sqrtf(h)) / a, y = baoa + t * bard;
        if (y > 0.0f && y < baba && t > 1e-4f) best = t;
    }
    float t1 = hit_sphere(ro, rd, pa, r), t2 = hit_sphere(ro, rd, pb, r);
    if (t1 > 0.0f && (best < 0.0f || t1 < best)) best = t1;
    if (t2 > 0.0f && (best < 0.0f || t2 < best)) best = t2;
    return best;
}

HD inline V3 capsule_normal(V3 p, V3 pa, V3 pb, float r) {
    V3 ba = pb - pa;
    float k = clampf(dot(p - pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return (1.0f / r) * (p - (pa + k * ba));
}

// flat rectangle: plane hit, then |local u| <= hu and |local v| <= hv (two-sided)
HD inline float hit_rect(V3 ro, V3 rd, const Panel& P) {
    float den = dot(rd, P.n);
    if (fabsf(den) < 1e-7f) return -1.0f;
    float t = dot(P.c - ro, P.n) / den;
    if (t <= 1e-4f) return -1.0f;
    V3 q = (ro + t * rd) - P.c;
    return (fabsf(dot(q, P.u)) <= P.hu && fabsf(dot(q, P.v)) <= P.hv) ? t : -1.0f;
}

HD inline float hit_terrain(const Terrain& T, V3 ro, V3 rd, float tmax) {
    float t = 0.05f, prev_t = t;
    if (ro.z + t * rd.z - height(T, ro.x + t * rd.x, ro.y + t * rd.y) < 0) return -1.0f;   // starting under ground
    while (t < tmax) {
        prev_t = t;
        t += 0.05f + 0.02f * t;
        V3 p = ro + t * rd;
        if (p.z - height(T, p.x, p.y) < 0.0f) {                     // crossed the surface: bisect
            float lo = prev_t, hi = t;
            for (int k = 0; k < 14; ++k) {
                float m = 0.5f * (lo + hi);
                V3 q = ro + m * rd;
                if (q.z - height(T, q.x, q.y) < 0.0f) hi = m; else lo = m;
            }
            return 0.5f * (lo + hi);
        }
    }
    return -1.0f;
}

HD inline Hit trace(const Scene& s, V3 ro, V3 rd, float tmax) {
    Hit h;
    h.t = tmax; h.id = SKY; h.n = mk(0, 0, 1); h.k = -1;
    float t = hit_terrain(s.T, ro, rd, tmax);
    if (t > 0 && t < h.t) { V3 p = ro + t * rd; h.t = t; h.id = GROUND; h.n = terrain_normal(s.T, p.x, p.y); }
    t = hit_sphere(ro, rd, s.hero.body, s.hero.body_r);
    if (t > 0 && t < h.t) { h.t = t; h.id = BODY; h.n = normalize((ro + t * rd) - s.hero.body); }
    t = hit_sphere(ro, rd, s.hero.head, s.hero.head_r);
    if (t > 0 && t < h.t) { h.t = t; h.id = HEAD; h.n = normalize((ro + t * rd) - s.hero.head); }
    t = hit_capsule(ro, rd, s.hero.sword_a, s.hero.sword_b, s.hero.sword_r);
    if (t > 0 && t < h.t) { h.t = t; h.id = SWORD; h.n = capsule_normal(ro + t * rd, s.hero.sword_a, s.hero.sword_b, s.hero.sword_r); }
    for (int k = 0; k < s.n_panels; ++k) {
        const Panel& P = s.panels[k];
        t = hit_rect(ro, rd, P);
        if (t > 0 && t < h.t) { h.t = t; h.id = PANEL; h.n = dot(rd, P.n) < 0 ? P.n : -1.0f * P.n; h.k = k; }
        V3 top = P.c - P.hv * P.v;                                  // post from the ground to the panel's lower edge
        t = hit_capsule(ro, rd, P.post_base, top, 0.04f);
        if (t > 0 && t < h.t) { h.t = t; h.id = POST; h.n = capsule_normal(ro + t * rd, P.post_base, top, 0.04f); h.k = k; }
    }
    if (s.droid.visible) {
        const Droid& D = s.droid;
        t = hit_capsule(ro, rd, D.base, D.head, 0.09f);                       // mast
        if (t > 0 && t < h.t) { h.t = t; h.id = DROID; h.n = capsule_normal(ro + t * rd, D.base, D.head, 0.09f); h.k = 0; }
        t = hit_sphere(ro, rd, D.head, D.head_r);                              // PTZ head
        if (t > 0 && t < h.t) { h.t = t; h.id = DROID; h.n = normalize((ro + t * rd) - D.head); h.k = 1; }
        if (D.laser_on) {
            t = hit_capsule(ro, rd, D.laser_a, D.laser_b, 0.03f);
            if (t > 0 && t < h.t) { h.t = t; h.id = LASER; h.n = mk(0, 0, 1); h.k = -1; }
        }
    }
    return h;
}

// any occluder between p and the sun? (shadow ray; the thin laser beam casts no shadow)
HD inline bool in_shadow(const Scene& s, V3 p, V3 n) {
    V3 o = p + 2e-3f * n;
    int id = trace(s, o, s.sun, 80.0f).id;
    return id != SKY && id != LASER;
}

// line of sight from a to b against the terrain only
HD inline bool terrain_blocks(const Terrain& T, V3 a, V3 b) {
    V3 d = b - a;
    float L = len(d);
    float t = hit_terrain(T, a, (1.0f / L) * d, L);
    return t > 0.0f && t < L - 1e-3f;
}

// ------------------------------------------------------------------ sky, lighting, shading
HD inline V3 horizon_color() { return mk(0.72f, 0.80f, 0.90f); }

HD inline V3 sky(const Scene& s, V3 rd) {
    float up = clampf(rd.z, 0.0f, 1.0f);
    V3 col = mix(horizon_color(), mk(0.20f, 0.42f, 0.82f), sqrtf(up));     // pale horizon -> deep zenith
    float mu = clampf(dot(rd, s.sun), 0.0f, 1.0f);
    float halo = powf(mu, 24.0f), glow = powf(mu, 4.0f);
    col = col + 0.18f * glow * mk(0.55f, 0.50f, 0.42f) + 0.35f * halo * mk(1.0f, 0.97f, 0.90f);
    if (mu > 0.9993f) col = mk(1.0f, 0.99f, 0.95f);                       // the sun's disk
    if (rd.z < 0.0f) col = mix(col, 0.92f * horizon_color(), clampf(-8.0f * rd.z, 0.0f, 1.0f));
    return mk(clampf(col.x, 0, 1), clampf(col.y, 0, 1), clampf(col.z, 0, 1));
}

HD inline V3 shade(const Scene& s, V3 ro, V3 rd, const Hit& h) {
    if (h.id == SKY) return sky(s, rd);
    if (h.id == LASER) return mk(0.45f, 1.0f, 0.55f);                     // 532 nm game laser: emissive green
    V3 p = ro + h.t * rd, albedo;
    float spec_k = 0.0f, shininess = 1.0f;
    if (h.id == DROID) {                                                   // brushed-metal droid, glowing lens
        V3 to_cam = normalize(ro - p);
        if (h.k == 1 && dot(h.n, normalize(s.droid.laser_b - s.droid.head)) > 0.85f && s.droid.laser_on) return mk(0.6f, 1.0f, 0.7f);
        albedo = h.k == 1 ? mk(0.82f, 0.84f, 0.88f) : mk(0.50f, 0.53f, 0.57f);
        spec_k = 0.5f;
        shininess = 60.0f;
        (void)to_cam;
    } else
    if (h.id == GROUND) {
        float k = clampf(0.5f + 0.4f * p.z, 0.0f, 1.0f);                    // higher ground: drier, browner
        albedo = mk(0.28f + 0.17f * k, 0.46f - 0.08f * k, 0.20f + 0.05f * k);
    } else if (h.id == BODY) albedo = mk(0.95f, 0.52f, 0.10f);             // orange tunic
    else if (h.id == HEAD) albedo = mk(0.90f, 0.72f, 0.56f);
    else if (h.id == SWORD) albedo = mk(1.00f, 0.92f, 0.30f);              // yellow foam sword
    else if (h.id == POST) { albedo = mk(0.62f, 0.64f, 0.66f); spec_k = 0.2f; shininess = 20.0f; }
    else {                                                                 // photovoltaic panel: cells, busbars, frame
        const Panel& P = s.panels[h.k];
        float a = dot(p - P.c, P.u) / P.hu, b = dot(p - P.c, P.v) / P.hv;
        float ga = (a + 1.0f) * 5.0f, gb = (b + 1.0f) * 3.0f;
        bool line = (ga - floorf(ga)) < 0.06f || (gb - floorf(gb)) < 0.08f;
        bool frame = fabsf(a) > 0.96f || fabsf(b) > 0.94f;
        albedo = frame ? mk(0.70f, 0.72f, 0.74f) : (line ? mk(0.55f, 0.58f, 0.62f) : mk(0.05f, 0.10f, 0.26f));
        if (!frame) albedo = mix(albedo, mk(0.50f, 0.44f, 0.34f), 0.75f * P.dirt);   // dust dulls the glass
        spec_k = frame ? 0.3f : 0.9f * (1.0f - P.dirt);
        shininess = 140.0f;
    }
    float sh = in_shadow(s, p, h.n) ? 0.0f : 1.0f;
    float lam = clampf(dot(h.n, s.sun), 0.0f, 1.0f) * sh;
    // near-neutral sky light + green ground bounce: a strongly blue ambient desaturated the shaded tunic below
    // the vision threshold (caught by test_scene3d "centre pixel sees the orange tunic")
    V3 ambient = (0.30f + 0.20f * h.n.z) * mk(0.66f, 0.70f, 0.78f) + (0.10f - 0.08f * h.n.z) * mk(0.46f, 0.50f, 0.36f);
    V3 col = mul(albedo, ambient + 0.95f * lam * mk(1.0f, 0.95f, 0.86f));
    if (spec_k > 0.0f && sh > 0.0f) {                                      // Blinn-Phong sun glint
        V3 hv = normalize(s.sun - rd);
        float spec = powf(clampf(dot(h.n, hv), 0.0f, 1.0f), shininess);
        col = col + (spec_k * spec) * mk(1.0f, 0.97f, 0.92f);
    }
    float haze = 1.0f - expf(-h.t / 120.0f);                               // aerial perspective (clear day)
    col = mix(col, horizon_color(), haze);
    return mk(clampf(col.x, 0, 1), clampf(col.y, 0, 1), clampf(col.z, 0, 1));
}

HD inline V3 render_pixel(const Scene& s, const Camera& c, int i, int j) {
    V3 rd = pixel_ray(c, i, j);
    return shade(s, c.pos, rd, trace(s, c.pos, rd, 80.0f));
}

// "orange-ness" feature for vision, as CHROMATICITY so it survives shading: (R - (G+B)/2) / (R+G+B).
// Tunic ~0.4 (lit or shaded), sword ~0.16, skin ~0.12, brown hilltops <= ~0.13, sun halo, glints, panels,
// grass and sky below that; the 0.15 offset keeps only the tunic, then the result is scaled to ~[0, 1].
HD inline float target_feature(V3 rgb) {
    float sum = rgb.x + rgb.y + rgb.z + 1e-3f;
    return clampf(((rgb.x - 0.5f * (rgb.y + rgb.z)) / sum - 0.15f) / 0.3f, 0.0f, 1.0f);
}

}  // namespace d3
