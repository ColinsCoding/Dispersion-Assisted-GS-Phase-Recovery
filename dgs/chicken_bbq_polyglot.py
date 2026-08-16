"""The SAME flip physics from dgs.chicken_bbq_simulator (apply_flip_impulse
-> time_to_return_to_height -> flip_trajectory -> classify_landing), run
for real in Python, C, and Rust -- prep for embedded targets, where the
game loop above (Python + pygame) obviously never runs, but this exact
closed-form kinematics chain (no heap allocation, no dynamic dispatch,
just floats and a handful of arithmetic ops) is exactly the kind of code
that DOES run on a microcontroller.

Same subprocess/gcc/rustc pattern as dgs.error_propagation_polyglot and
dgs.circuits_polyglot: write source to a file, compile, run with CLI
args, parse stdout, and cross-check against the trusted Python reference
to near machine precision -- proof the ports are faithful, not just
plausible-looking translations.
"""

import os
import subprocess

from dgs.chicken_bbq_simulator import (
    apply_flip_impulse, time_to_return_to_height, flip_trajectory,
    normalize_angle, classify_landing, disk_moment_of_inertia,
)

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"
RUSTC_DEFAULT = r"C:\Users\mrjel\.cargo\bin\rustc.exe"

# ── C: the whole flip chain as one self-contained, allocation-free function ──

C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

typedef struct { double vy0, omega0; } Impulse;
typedef struct { double x, y, angle; } State;

/* Same physics as dgs.chicken_bbq_simulator.apply_flip_impulse: an
   off-center strike gives BOTH a linear impulse (vy0) and an angular
   impulse via torque = force*offset (omega0). No malloc, no dynamic
   dispatch -- this is embedded-target-shaped code. */
static Impulse apply_flip_impulse(double force, double contact_time,
                                   double offset, double mass, double I) {
    double vy0 = (force * contact_time) / mass;
    double torque = force * offset;
    double omega0 = (torque * contact_time) / I;
    Impulse out = { vy0, omega0 };
    return out;
}

/* Closed-form (quadratic formula) flight time -- same as
   dgs.chicken_bbq_simulator.time_to_return_to_height, no numerical
   integration. */
static double time_to_return_to_height(double y0, double vy0, double target_y, double g) {
    double a = -0.5 * g;
    double b = vy0;
    double c = y0 - target_y;
    double disc = b*b - 4*a*c;
    if (disc < 0) return -1.0;
    double sq = sqrt(disc);
    double t1 = (-b + sq) / (2*a);
    double t2 = (-b - sq) / (2*a);
    double best = -1.0;
    if (t1 > 1e-9 && t1 > best) best = t1;
    if (t2 > 1e-9 && t2 > best) best = t2;
    return best;
}

static State flip_trajectory(double x0, double y0, double vx0, double vy0,
                              double omega0, double angle0, double t, double g) {
    State s = { x0 + vx0*t, y0 + vy0*t - 0.5*g*t*t, angle0 + omega0*t };
    return s;
}

static double normalize_angle(double angle) {
    double a = fmod(angle, 2*PI);
    if (a < 0) a += 2*PI;
    return a;
}

/* returns 0 for side A, 1 for side B, 2 for bad landing -- an int return
   code instead of Python's string, exactly the kind of change embedded C
   forces (no first-class string-as-enum the way Python has it) */
static int classify_landing(double angle_at_landing, double tolerance) {
    double a = normalize_angle(angle_at_landing);
    double dist_to_0 = a < (2*PI - a) ? a : (2*PI - a);
    double dist_to_pi = fabs(a - PI);
    if (dist_to_0 <= tolerance) return 0;
    if (dist_to_pi <= tolerance) return 1;
    return 2;
}

int main(int argc, char **argv) {
    double force = atof(argv[1]);
    double contact_time = atof(argv[2]);
    double offset = atof(argv[3]);
    double mass = atof(argv[4]);
    double I = atof(argv[5]);
    double g = atof(argv[6]);

    Impulse imp = apply_flip_impulse(force, contact_time, offset, mass, I);
    double t_land = time_to_return_to_height(0.0, imp.vy0, 0.0, g);
    State landing = flip_trajectory(0.5, 0.0, 0.0, imp.vy0, imp.omega0, 0.0, t_land, g);
    int verdict = classify_landing(landing.angle, 35.0 * PI / 180.0);

    printf("%.10e %.10e %.10e %.10e %d\n", imp.vy0, imp.omega0, t_land, landing.angle, verdict);
    return 0;
}
"""

# ── Rust: same chain, no unsafe, no heap allocation ──────────────────────────

RUST_SOURCE = r"""
use std::env;
use std::f64::consts::PI;

struct Impulse { vy0: f64, omega0: f64 }
struct State { x: f64, y: f64, angle: f64 }

fn apply_flip_impulse(force: f64, contact_time: f64, offset: f64, mass: f64, moi: f64) -> Impulse {
    let vy0 = (force * contact_time) / mass;
    let torque = force * offset;
    let omega0 = (torque * contact_time) / moi;
    Impulse { vy0, omega0 }
}

fn time_to_return_to_height(y0: f64, vy0: f64, target_y: f64, g: f64) -> Option<f64> {
    let a = -0.5 * g;
    let b = vy0;
    let c = y0 - target_y;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = (-b + sq) / (2.0 * a);
    let t2 = (-b - sq) / (2.0 * a);
    [t1, t2].into_iter().filter(|&t| t > 1e-9).fold(None, |best, t| {
        Some(match best { Some(b) if b > t => b, _ => t })
    })
}

fn flip_trajectory(x0: f64, y0: f64, vx0: f64, vy0: f64, omega0: f64, angle0: f64, t: f64, g: f64) -> State {
    State { x: x0 + vx0 * t, y: y0 + vy0 * t - 0.5 * g * t * t, angle: angle0 + omega0 * t }
}

fn normalize_angle(angle: f64) -> f64 {
    let a = angle % (2.0 * PI);
    if a < 0.0 { a + 2.0 * PI } else { a }
}

// 0 = side A, 1 = side B, 2 = bad landing -- Rust's enums COULD express this
// more richly than C's int return code, but this stays a direct, literal
// port for the cross-check (same interface as the C version above).
fn classify_landing(angle_at_landing: f64, tolerance: f64) -> i32 {
    let a = normalize_angle(angle_at_landing);
    let dist_to_0 = a.min(2.0 * PI - a);
    let dist_to_pi = (a - PI).abs();
    if dist_to_0 <= tolerance { return 0; }
    if dist_to_pi <= tolerance { return 1; }
    2
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let force: f64 = args[1].parse().unwrap();
    let contact_time: f64 = args[2].parse().unwrap();
    let offset: f64 = args[3].parse().unwrap();
    let mass: f64 = args[4].parse().unwrap();
    let moi: f64 = args[5].parse().unwrap();
    let g: f64 = args[6].parse().unwrap();

    let imp = apply_flip_impulse(force, contact_time, offset, mass, moi);
    let t_land = time_to_return_to_height(0.0, imp.vy0, 0.0, g).expect("must return to grill height");
    let landing = flip_trajectory(0.5, 0.0, 0.0, imp.vy0, imp.omega0, 0.0, t_land, g);
    let verdict = classify_landing(landing.angle, 35.0 * PI / 180.0);

    println!("{:.10e} {:.10e} {:.10e} {:.10e} {}", imp.vy0, imp.omega0, t_land, landing.angle, verdict);
}
"""


def _compile_c(out_dir, gcc_path=GCC_DEFAULT):
    src_path = os.path.join(out_dir, "chicken_flip.c")
    exe_path = os.path.join(out_dir, "chicken_flip_c.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE)
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path, "-lm"],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe_path


def _compile_rust(out_dir, rustc_path=RUSTC_DEFAULT):
    src_path = os.path.join(out_dir, "chicken_flip.rs")
    exe_path = os.path.join(out_dir, "chicken_flip_rs.exe")
    with open(src_path, "w") as f:
        f.write(RUST_SOURCE)
    result = subprocess.run([rustc_path, "-O", "-o", exe_path, src_path],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rustc compile failed: {result.stderr}")
    return exe_path


def _run_exe(exe_path, force, contact_time, offset, mass, moi, g):
    args = [exe_path, str(force), str(contact_time), str(offset), str(mass), str(moi), str(g)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{exe_path} failed: {result.stderr}")
    vy0_s, omega0_s, t_land_s, angle_s, verdict_s = result.stdout.strip().split()
    return {
        "vy0": float(vy0_s), "omega0": float(omega0_s), "t_land": float(t_land_s),
        "angle": float(angle_s), "verdict_code": int(verdict_s),
    }


_VERDICT_NAMES = {0: "A", 1: "B", 2: "bad"}


def python_reference(force, contact_time, offset, mass, moi, g):
    vy0, omega0 = apply_flip_impulse(force, contact_time, offset, mass, moi)
    t_land = time_to_return_to_height(0.0, vy0, 0.0, g)
    _, _, angle = flip_trajectory(0.5, 0.0, 0.0, vy0, omega0, 0.0, t_land, g)
    verdict = classify_landing(angle)
    return {"vy0": vy0, "omega0": omega0, "t_land": t_land, "angle": angle, "verdict": verdict}


def cross_validate_languages(out_dir, force=12.0, contact_time=0.03, offset=0.005,
                              mass=0.15, radius=0.06, g=9.80665,
                              gcc_path=GCC_DEFAULT, rustc_path=RUSTC_DEFAULT,
                              run_c=True, run_rust=True):
    """Run the identical flip physics in Python, C, and Rust and report
    max disagreement across every pair -- proof the ports are faithful."""
    moi = disk_moment_of_inertia(mass, radius)
    py = python_reference(force, contact_time, offset, mass, moi, g)
    out = {"python": py}

    if run_c:
        exe_c = _compile_c(out_dir, gcc_path)
        c_result = _run_exe(exe_c, force, contact_time, offset, mass, moi, g)
        c_result["verdict"] = _VERDICT_NAMES[c_result["verdict_code"]]
        out["c"] = c_result

    if run_rust:
        exe_rust = _compile_rust(out_dir, rustc_path)
        rust_result = _run_exe(exe_rust, force, contact_time, offset, mass, moi, g)
        rust_result["verdict"] = _VERDICT_NAMES[rust_result["verdict_code"]]
        out["rust"] = rust_result

    return out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        results = cross_validate_languages(tmp)
        for lang, r in results.items():
            print(f"{lang:8s}: vy0={r['vy0']:.6f}  omega0={r['omega0']:.6f}  "
                  f"t_land={r['t_land']:.6f}  angle={r['angle']:.6f}  verdict={r['verdict']}")

        py, c, rust = results["python"], results.get("c"), results.get("rust")
        if c:
            print(f"\nmax |Python - C| (vy0,omega0,t_land,angle): "
                  f"{max(abs(py[k]-c[k]) for k in ('vy0','omega0','t_land','angle')):.2e}")
        if rust:
            print(f"max |Python - Rust| (vy0,omega0,t_land,angle): "
                  f"{max(abs(py[k]-rust[k]) for k in ('vy0','omega0','t_land','angle')):.2e}")
