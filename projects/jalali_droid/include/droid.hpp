// droid.hpp -- a Jalali-themed PTZ camera droid (think Zelda Guardian, but it only tags you with a
// harmless game laser). One header for the whole C++17/CUDA project:
//
//   physics.cpp     unit vectors and PTZ aim; DC gimbal motor (mechanics + E&M) as an ODE integrated
//                   with RK4; vibration-isolated camera head; Gaussian laser beam (photonics)
//   vision_cpu.cpp  synthetic camera frame, radix-2 FFT, Phase Stretch Transform (Jalali lab), target
//                   locator
//   vision_gpu.cu   the same PST on the GPU with cuFFT (compiled only when DROID_CUDA is defined)
//   main.cpp        command-line game loop, ASCII animation, CSV log, exit codes for $?
#pragma once
#include <cmath>
#include <complex>
#include <string>
#include <vector>

namespace droid {

constexpr double PI = 3.14159265358979323846;

// ------------------------------------------------------------------ vectors
struct Vec3 { double x, y, z; };
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

Vec3 unit(Vec3 a);                                   // a / |a|; throws std::invalid_argument for |a| = 0
Vec3 aim_vector(double pan, double tilt);            // unit vector (cos t cos p, cos t sin p, sin t)
void pan_tilt_of(Vec3 d, double& pan, double& tilt); // inverse of aim_vector
double wrap_pi(double a);                            // angle into (-pi, pi]

// ------------------------------------------------------------------ DC gimbal motor (one axis)
// Mechanics:  J w' = Kt i - b w          Electromagnetism:  L i' = V - R i - Ke w      theta' = w
// Controller: V = clamp(Kp (theta_ref - theta) - Kd w, -Vmax, Vmax)
struct MotorParams { double R = 2.0, L = 1e-3, Kt = 0.05, Ke = 0.05, J = 2e-4, b = 1e-4, Vmax = 12.0, Kp = 30.0, Kd = 1.5; };
struct MotorState { double theta = 0, omega = 0, i = 0; };
MotorState motor_step(const MotorParams& p, MotorState s, double theta_ref, double dt);               // one RK4 step
MotorState motor_advance(const MotorParams& p, MotorState s, double theta_ref, double T, double dt);  // many steps

// ------------------------------------------------------------------ vibration-isolated camera head
// The walking droid rocks its base by u(t) = A sin(2 pi f t) (radians). The head sits on a torsional
// isolation mount: x'' = -2 zeta w0 (x' - u') - w0^2 (x - u). Rigid mount: x = u.
struct MountParams { double f0 = 3.0, zeta = 0.1; };
struct MountState { double x = 0, v = 0; };
MountState mount_step(const MountParams& p, MountState s, double t, double A, double f, double dt);
double transmissibility(double r, double zeta);      // |X/U| at r = f / f0

// ------------------------------------------------------------------ laser (Gaussian beam)
double rayleigh_range(double w0, double lambda);     // pi w0^2 / lambda
double beam_radius(double w0, double lambda, double z);

// ------------------------------------------------------------------ vision
struct Image { int n = 0; std::vector<float> px; };  // n x n, row-major, values ~[0, 1]
Image render_scene(int n, double err_pan, double err_tilt, double fov, double target_size, double sword_phase, unsigned seed);
void fft2_cpu(std::vector<std::complex<double>>& a, int n, bool inverse); // n must be a power of two
std::vector<double> pst_kernel(int n, double S, double W, double sigma_lp, std::vector<double>& lowpass);
std::vector<float> pst_cpu(const Image& img, double S, double W, double sigma_lp);
bool locate_target(const std::vector<float>& edge, int n, double thresh, double& cx, double& cy);

#ifdef DROID_CUDA
bool pst_gpu(const Image& img, double S, double W, double sigma_lp, std::vector<float>& out, std::string& err);
#endif

}  // namespace droid
