// physics.cpp -- unit vectors, the gimbal motor ODE (mechanics + E&M), the isolation mount, the laser beam.
#include "droid.hpp"

#include <algorithm>
#include <stdexcept>

namespace droid {

Vec3 unit(Vec3 a) {
    double n = norm(a);
    if (!(n > 0.0)) throw std::invalid_argument("unit(): zero-length vector has no direction");
    return (1.0 / n) * a;
}

Vec3 aim_vector(double pan, double tilt) {
    return {std::cos(tilt) * std::cos(pan), std::cos(tilt) * std::sin(pan), std::sin(tilt)};
}

void pan_tilt_of(Vec3 d, double& pan, double& tilt) {
    Vec3 u = unit(d);
    pan = std::atan2(u.y, u.x);
    tilt = std::asin(std::max(-1.0, std::min(1.0, u.z)));
}

double wrap_pi(double a) {
    a = std::fmod(a + PI, 2 * PI);
    if (a < 0) a += 2 * PI;
    return a - PI;
}

// ------------------------------------------------------------------ motor
namespace {
struct MotorDeriv { double dtheta, domega, di; };

MotorDeriv motor_rhs(const MotorParams& p, const MotorState& s, double ref) {
    double V = p.Kp * wrap_pi(ref - s.theta) - p.Kd * s.omega;
    V = std::max(-p.Vmax, std::min(p.Vmax, V));
    return {s.omega, (p.Kt * s.i - p.b * s.omega) / p.J, (V - p.R * s.i - p.Ke * s.omega) / p.L};
}

MotorState add(const MotorState& s, const MotorDeriv& k, double h) {
    return {s.theta + h * k.dtheta, s.omega + h * k.domega, s.i + h * k.di};
}
}  // namespace

MotorState motor_step(const MotorParams& p, MotorState s, double ref, double dt) {
    MotorDeriv k1 = motor_rhs(p, s, ref);
    MotorDeriv k2 = motor_rhs(p, add(s, k1, dt / 2), ref);
    MotorDeriv k3 = motor_rhs(p, add(s, k2, dt / 2), ref);
    MotorDeriv k4 = motor_rhs(p, add(s, k3, dt), ref);
    return {s.theta + dt / 6 * (k1.dtheta + 2 * k2.dtheta + 2 * k3.dtheta + k4.dtheta),
            s.omega + dt / 6 * (k1.domega + 2 * k2.domega + 2 * k3.domega + k4.domega),
            s.i + dt / 6 * (k1.di + 2 * k2.di + 2 * k3.di + k4.di)};
}

MotorState motor_advance(const MotorParams& p, MotorState s, double ref, double T, double dt) {
    if (!(dt > 0) || T < 0) throw std::invalid_argument("motor_advance: need dt > 0, T >= 0");
    int n = static_cast<int>(std::ceil(T / dt - 1e-9));
    double h = n > 0 ? T / n : 0.0;
    for (int k = 0; k < n; ++k) s = motor_step(p, s, ref, h);
    return s;
}

// ------------------------------------------------------------------ isolation mount
namespace {
void mount_rhs(const MountParams& p, double x, double v, double t, double A, double f, double& dx, double& dv) {
    double w0 = 2 * PI * p.f0, W = 2 * PI * f;
    double u = A * std::sin(W * t), ud = A * W * std::cos(W * t);
    dx = v;
    dv = -2 * p.zeta * w0 * (v - ud) - w0 * w0 * (x - u);
}
}  // namespace

MountState mount_step(const MountParams& p, MountState s, double t, double A, double f, double dt) {
    double a1, b1, a2, b2, a3, b3, a4, b4;
    mount_rhs(p, s.x, s.v, t, A, f, a1, b1);
    mount_rhs(p, s.x + dt / 2 * a1, s.v + dt / 2 * b1, t + dt / 2, A, f, a2, b2);
    mount_rhs(p, s.x + dt / 2 * a2, s.v + dt / 2 * b2, t + dt / 2, A, f, a3, b3);
    mount_rhs(p, s.x + dt * a3, s.v + dt * b3, t + dt, A, f, a4, b4);
    return {s.x + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4), s.v + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)};
}

double transmissibility(double r, double zeta) {
    double num = 1 + (2 * zeta * r) * (2 * zeta * r);
    double den = (1 - r * r) * (1 - r * r) + (2 * zeta * r) * (2 * zeta * r);
    return std::sqrt(num / den);
}

// ------------------------------------------------------------------ Gaussian beam
double rayleigh_range(double w0, double lambda) { return PI * w0 * w0 / lambda; }

double beam_radius(double w0, double lambda, double z) {
    double zr = rayleigh_range(w0, lambda);
    return w0 * std::sqrt(1 + (z / zr) * (z / zr));
}

}  // namespace droid
