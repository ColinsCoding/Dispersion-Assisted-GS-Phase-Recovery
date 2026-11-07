// game3d.hpp -- the 3D battle: character definitions (assets/characters.def), monster-battle damage,
// and the CPU / GPU renderers of the droid's camera view.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "scene3d.hpp"

namespace g3 {

// ------------------------------------------------------------------ definitions (INI-style, UTF-8)
using Section = std::map<std::string, std::string>;
using Defs = std::map<std::string, Section>;
bool load_defs(const std::string& path, Defs& out, std::string& err);
double num(const Defs& d, const std::string& sec, const std::string& key);       // throws if missing
std::string str(const Defs& d, const std::string& sec, const std::string& key);

struct Fighter {
    std::string name_ja, name_en, type, move_ja, move_en, move_type;
    int level = 1, hp = 1, max_hp = 1, attack = 1, defense = 1, speed = 1, move_power = 1;
};
Fighter fighter_from(const Defs& d, const std::string& sec);

// type multiplier attacker move type -> defender type ("A/B" dual types multiply)
double type_multiplier(const Defs& d, const std::string& move_type, const std::string& defender_type);
// floor(floor(floor(2L/5 + 2) * P * A / D) / 50) + 2, then x STAB 1.5, x type, x random in [0.85, 1]; at least 1
int damage(const Fighter& att, const Fighter& def, double type_mult, double rand01);

// ------------------------------------------------------------------ solar farm (photovoltaics)
// Lay out [solar] rows x cols panels tilted at the site latitude, facing the sun's azimuth; returns the count.
int build_solar_field(const Defs& d, d3::Scene& s);
// Power of panel k (W): eta * A * (1 - dirt/2) * (DNI * cos(theta) * unshaded + DHI * (1 + n_z) / 2), with the unshaded
// fraction from shadow rays at 3 x 3 points on the panel (hills, the hero, other panels can shade it).
double panel_power(const d3::Scene& s, int k, double dni, double dhi, double eta, double& unshaded);

// ------------------------------------------------------------------ rendering
void render_cpu(const d3::Scene& s, const d3::Camera& c, std::vector<float>& rgb);   // n*n*3
#ifdef DROID_CUDA
bool render_gpu(const d3::Scene& s, const d3::Camera& c, std::vector<float>& rgb, std::string& err);
#endif

}  // namespace g3
