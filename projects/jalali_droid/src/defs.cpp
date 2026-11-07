// defs.cpp -- read assets/characters.def and do the battle arithmetic.
#include "game3d.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace g3 {

namespace {
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, sep)) out.push_back(trim(part));
    return out;
}
}  // namespace

bool load_defs(const std::string& path, Defs& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::string line, sec;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (lineno == 1 && line.size() >= 3 && (unsigned char)line[0] == 0xEF) line = line.substr(3);  // UTF-8 BOM
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        if (line.front() == '[' && line.back() == ']') { sec = trim(line.substr(1, line.size() - 2)); out[sec]; continue; }
        size_t eq = line.find('=');
        if (eq == std::string::npos || sec.empty()) { err = path + ":" + std::to_string(lineno) + ": expected key = value"; return false; }
        out[sec][trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return true;
}

std::string str(const Defs& d, const std::string& sec, const std::string& key) {
    auto s = d.find(sec);
    if (s == d.end() || s->second.find(key) == s->second.end()) throw std::runtime_error("missing [" + sec + "] " + key);
    return s->second.at(key);
}

double num(const Defs& d, const std::string& sec, const std::string& key) { return std::stod(str(d, sec, key)); }

Fighter fighter_from(const Defs& d, const std::string& sec) {
    Fighter f;
    f.name_ja = str(d, sec, "name_ja");
    f.name_en = str(d, sec, "name_en");
    f.type = str(d, sec, "type");
    f.move_ja = str(d, sec, "move_ja");
    f.move_en = str(d, sec, "move_en");
    f.move_type = str(d, sec, "move_type");
    f.level = static_cast<int>(num(d, sec, "level"));
    f.hp = f.max_hp = static_cast<int>(num(d, sec, "hp"));
    f.attack = static_cast<int>(num(d, sec, "attack"));
    f.defense = static_cast<int>(num(d, sec, "defense"));
    f.speed = static_cast<int>(num(d, sec, "speed"));
    f.move_power = static_cast<int>(num(d, sec, "move_power"));
    if (f.level < 1 || f.hp < 1 || f.attack < 1 || f.defense < 1 || f.move_power < 1) throw std::runtime_error("[" + sec + "] stats must be >= 1");
    return f;
}

double type_multiplier(const Defs& d, const std::string& move_type, const std::string& defender_type) {
    double m = 1.0;
    auto types = d.find("types");
    for (const std::string& t : split(defender_type, '/')) {
        if (types == d.end()) break;
        auto it = types->second.find(move_type + ">" + t);
        if (it != types->second.end()) m *= std::stod(it->second);
    }
    return m;
}

int damage(const Fighter& att, const Fighter& def, double type_mult, double rand01) {
    int base = (2 * att.level) / 5 + 2;
    base = (base * att.move_power * att.attack) / def.defense;
    base = base / 50 + 2;
    bool stab = false;
    for (const std::string& t : split(att.type, '/')) stab = stab || (t == att.move_type);
    double r = 0.85 + 0.15 * rand01;
    int dmg = static_cast<int>(std::floor(base * (stab ? 1.5 : 1.0) * type_mult * r));
    return dmg < 1 ? 1 : dmg;
}

// ------------------------------------------------------------------ solar farm
int build_solar_field(const Defs& d, d3::Scene& s) {
    int rows = static_cast<int>(num(d, "solar", "rows")), cols = static_cast<int>(num(d, "solar", "cols"));
    if (rows < 1 || cols < 1 || rows * cols > d3::MAX_PANELS) throw std::runtime_error("[solar] rows * cols must be 1.." + std::to_string(d3::MAX_PANELS));
    float tilt = static_cast<float>(num(d, "solar", "latitude_deg") * 3.14159265358979 / 180.0);
    float hu = static_cast<float>(0.5 * num(d, "solar", "panel_w_m")), hv = static_cast<float>(0.5 * num(d, "solar", "panel_h_m"));
    float post = static_cast<float>(num(d, "solar", "post_height_m"));
    float fx = static_cast<float>(num(d, "solar", "field_x_m")), fy = static_cast<float>(num(d, "solar", "field_y_m"));
    float rs = static_cast<float>(num(d, "solar", "row_spacing_m")), cs = static_cast<float>(num(d, "solar", "col_spacing_m"));
    d3::V3 hs = d3::normalize(d3::mk(s.sun.x, s.sun.y, 0.0f));              // the sun's azimuth (horizontal)
    d3::V3 z = d3::mk(0, 0, 1);
    d3::V3 n = d3::normalize(std::sin(tilt) * hs + std::cos(tilt) * z);     // tilted up by the latitude, toward the sun
    d3::V3 u = d3::normalize(d3::cross(z, n));                              // along the row (horizontal)
    d3::V3 v = d3::cross(n, u);                                             // up the slope of the panel
    s.n_panels = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            d3::V3 g = d3::mk(fx, fy, 0) + (static_cast<float>(r) - 0.5f * (rows - 1)) * rs * hs + (static_cast<float>(c) - 0.5f * (cols - 1)) * cs * u;
            d3::V3 lower = d3::mk(g.x, g.y, d3::height(s.T, g.x, g.y) + post);
            d3::Panel& P = s.panels[s.n_panels++];
            P.c = lower + hv * v;
            P.n = n; P.u = u; P.v = v; P.hu = hu; P.hv = hv;
            P.post_base = d3::mk(g.x, g.y, d3::height(s.T, g.x, g.y));
        }
    return s.n_panels;
}

double panel_power(const d3::Scene& s, int k, double dni, double dhi, double eta, double& unshaded) {
    const d3::Panel& P = s.panels[k];
    int lit = 0;
    for (int a = -1; a <= 1; ++a)
        for (int b = -1; b <= 1; ++b) {
            d3::V3 q = P.c + (0.8f * a * P.hu) * P.u + (0.8f * b * P.hv) * P.v;
            lit += d3::in_shadow(s, q, P.n) ? 0 : 1;
        }
    unshaded = lit / 9.0;
    double cos_theta = std::max(0.0, static_cast<double>(d3::dot(P.n, s.sun)));
    double area = 4.0 * P.hu * P.hv;
    double soiling = 1.0 - 0.5 * P.dirt;                                   // fully dusty glass passes half the light
    return eta * area * soiling * (dni * cos_theta * unshaded + dhi * 0.5 * (1.0 + P.n.z));
}

// ------------------------------------------------------------------ CPU renderer (same HD code as the GPU)
void render_cpu(const d3::Scene& s, const d3::Camera& c, std::vector<float>& rgb) {
    rgb.resize(static_cast<size_t>(c.n) * c.n * 3);
    for (int j = 0; j < c.n; ++j)
        for (int i = 0; i < c.n; ++i) {
            d3::V3 p = d3::render_pixel(s, c, i, j);
            size_t k = (static_cast<size_t>(j) * c.n + i) * 3;
            rgb[k] = p.x; rgb[k + 1] = p.y; rgb[k + 2] = p.z;
        }
}

}  // namespace g3
