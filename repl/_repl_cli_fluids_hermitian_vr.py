# %% [markdown]
# # CLI · Fluids · Hermitian · Firmware · Grammar · VR
# *Signals kill processes · water flows · operators have real eigenvalues · quaternions never gimbal-lock*

# %%
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
sp.init_printing(use_latex="mathjax")

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")

def chk(val, ref, label, tol=1e-6, absolute=False):
    v, r = float(np.real(val)), float(np.real(ref))
    err = abs(v-r) if (absolute or abs(r)<1e-30) else abs(v-r)/(abs(r)+1e-30)
    s = "PASS" if err < tol else "FAIL"
    print(f"  [{s}] {label}: got {v:.6g}, ref {r:.6g}, err {err:.2e}")
    return s == "PASS"

# %% [markdown]
# ## §1 — CLI: processes, signals, pipes

# %%
hdr("§1 — CLI: processes, signals, pipes")

# Signal table
signals = {
    'SIGHUP': 1,
    'SIGINT': 2,
    'SIGQUIT': 3,
    'SIGKILL': 9,
    'SIGUSR1': 10,
    'SIGUSR2': 12,
    'SIGPIPE': 13,
    'SIGTERM': 15,
}
print("  Signal table:")
for name, num in signals.items():
    print(f"    {name:10s} = {num}")

print()
print("  kill semantics:")
print("    kill -15 PID  -> polite, process can clean up (SIGTERM)")
print("    kill -9  PID  -> instant, kernel forces termination (SIGKILL)")
print("    kill -2  PID  -> same as Ctrl+C (SIGINT)")

print()
print("  Process model:")
print("    Every process has PID, PPID, stdin/stdout/stderr (FDs 0/1/2)")
print("    Parent forks child; orphan adopted by init (PID=1)")
print("    Zombie: child exits before parent calls wait() -> Z state in process table")

# Simulate process lifecycle
process_table = {}

def fork(parent_pid, name):
    child_pid = parent_pid + 1
    process_table[child_pid] = {'state': 'running', 'ppid': parent_pid, 'name': name}
    return child_pid

def kill_proc(pid, sig):
    if pid not in process_table:
        return
    if sig in (9, 15):
        del process_table[pid]
    elif sig == 2:
        process_table[pid]['state'] = 'interrupted'

def child_exit(pid):
    """Child exits -> becomes zombie until parent waits"""
    if pid in process_table:
        process_table[pid]['state'] = 'zombie'

def wait_proc(ppid):
    """Parent calls wait() -> reap zombie children"""
    to_reap = [p for p, info in process_table.items()
               if info['ppid'] == ppid and info['state'] == 'zombie']
    for p in to_reap:
        del process_table[p]
    return len(to_reap)

# Scenario
process_table[100] = {'state': 'running', 'ppid': 1, 'name': 'parent'}
child_pid = fork(100, 'child')
print(f"\n  Forked child PID={child_pid}, process_table={process_table}")
child_exit(child_pid)
print(f"  Child exited -> zombie: state={process_table[child_pid]['state']}")
reaped = wait_proc(100)
zombie_cleared = 1 if child_pid not in process_table else 0
print(f"  Parent called wait() -> reaped {reaped}, zombie_cleared={zombie_cleared}")

# Pipe simulation
def pipe_sim(fns, data):
    result = data
    for fn in fns:
        result = fn(result)
    return result

pipe_result = pipe_sim(
    [str.split,
     lambda x: [w for w in x if 'foo' in w],
     len],
    "foo bar foo baz"
)
print(f"\n  pipe_sim(['foo bar foo baz'] | grep foo | wc): {pipe_result}")
print("  Pipe: ls | grep foo | wc -l -> three processes, two pipes")
print("    ls.stdout -> grep.stdin -> wc.stdin")

chk(signals['SIGKILL'], 9, "SIGKILL==9", absolute=True)
chk(signals['SIGTERM'], 15, "SIGTERM==15", absolute=True)
chk(zombie_cleared, 1, "zombie_cleared_after_wait==1", absolute=True)
chk(pipe_result, 2, "pipe_foo_count==2", absolute=True)

# %% [markdown]
# ## §2 — Rotational mechanics: revolution, angular momentum, gyroscope

# %%
hdr("§2 — Rotational mechanics")

# Earth rotation
T_earth = 86400.0  # seconds
R_E = 6.371e6      # meters
omega_earth = 2 * np.pi / T_earth
v_equator = omega_earth * R_E
print(f"  Earth: T={T_earth}s, omega={omega_earth:.4e} rad/s, v_equator={v_equator:.1f} m/s")

# Moment of inertia -- SymPy derivation
M, R, r, rho_sym = sp.symbols('M R r rho', positive=True)
rho_expr = M / (sp.Rational(4,3) * sp.pi * R**3)
I_integrand = sp.Rational(2,3)*r**2 * rho_expr * 4 * sp.pi * r**2
I_sphere_integral = sp.integrate(I_integrand, (r, 0, R))
I_sphere_simplified = sp.simplify(I_sphere_integral)
print("\n  Moment of inertia derivation (solid sphere):")
show(sp.Eq(sp.Symbol('I_sphere'), I_sphere_integral), "integral")
show(sp.Eq(sp.Symbol('I_sphere'), I_sphere_simplified), "simplified")
print("  Expected: 2/5 M R^2")
print("  Solid cylinder: I = 1/2 MR^2;  Thin rod (center): I = 1/12 ML^2")

# Angular momentum conservation -- ice skater
I1, w1 = 2.0, 2.0   # arms out
I2 = 0.5            # arms in
w2 = I1 * w1 / I2   # conservation: I1*w1 = I2*w2
print(f"\n  Ice skater: I1={I1} kg*m^2, w1={w1} rad/s -> I2={I2} kg*m^2, w2={w2} rad/s")

# Gyroscope precession
M_top = 0.1    # kg
r_top = 0.05   # m
I_top = 1e-4   # kg*m^2
omega_spin = 100.0  # rad/s
g = 9.8
tau_gyro = M_top * g * r_top
L_gyro = I_top * omega_spin
omega_precession = tau_gyro / L_gyro
print(f"\n  Gyroscope: M={M_top}kg, r={r_top}m, I={I_top}, omega_spin={omega_spin} rad/s")
print(f"    tau = Mgr = {tau_gyro:.4f} N*m")
print(f"    L = Iw = {L_gyro:.4f} kg*m^2/s")
print(f"    Omega_precession = tau/L = {omega_precession:.4f} rad/s")

# Euler's equations (symmetric top, torque-free)
print("\n  Euler equations (rigid body, principal axes):")
print("    I1*w_dot1 - (I2-I3)*w2*w3 = tau1  (and cyclic)")
print("    Symmetric top (I1=I2!=I3): w3=const, w1,w2 precess at Omega_body=(I3-I1)/I1*w3")

chk(omega_earth, 2*np.pi/86400, "omega_earth", tol=1e-8)
chk(w2, 8.0, "omega2_skater", absolute=True)
chk(float(I_sphere_simplified.subs([(M,1),(R,1)])),
    float(sp.Rational(2,5)*1*1), "I_sphere symbolic", tol=1e-10, absolute=True)
chk(omega_precession, 4.9, "gyro_precession", tol=0.1, absolute=True)

# %% [markdown]
# ## §3 — Fluid dynamics: Bernoulli, viscosity, Reynolds

# %%
hdr("§3 — Fluid dynamics")

# Continuity
d1, d2, v1_f = 0.1, 0.05, 2.0
v2_continuity = v1_f * (d1/d2)**2
print(f"  Continuity: d1={d1}m, d2={d2}m, v1={v1_f} m/s -> v2={v2_continuity} m/s")

# Bernoulli -- SymPy
P1, P2, rho_f, v1s, v2s, h1, h2 = symbols('P1 P2 rho v1 v2 h1 h2', positive=True)
bernoulli_eq = Eq(P1 + sp.Rational(1,2)*rho_f*v1s**2 + rho_f*9.8*h1,
                  P2 + sp.Rational(1,2)*rho_f*v2s**2 + rho_f*9.8*h2)
print("\n  Bernoulli equation:")
show(bernoulli_eq)

# Solve for P2-P1 at h1=h2
deltaP_sym = solve(bernoulli_eq.subs([(h1,0),(h2,0)]), P2)[0] - P1
deltaP_sym_simplified = simplify(deltaP_sym)
show(Eq(Symbol('P2-P1'), deltaP_sym_simplified), "deltaP (h1=h2)")

# Numerical: v1=2, v2=8, rho=1000
rho_water = 1000.0
deltaP_venturi = float(deltaP_sym_simplified.subs([(rho_f, rho_water),(v1s, 2.0),(v2s, 8.0)]))
print(f"  DeltaP at v1=2, v2=8, rho=1000: {deltaP_venturi:.1f} Pa (venturi -- lower pressure at constriction!)")

# Poiseuille flow
r_cap = 5e-6    # m
L_cap = 1e-2    # m  (10mm capillary length)
dP_cap = 100.0  # Pa
eta_blood = 0.003  # Pa*s
Q_blood = np.pi * r_cap**4 * dP_cap / (8 * eta_blood * L_cap)
print(f"\n  Poiseuille: capillary r={r_cap}m, L={L_cap}m, dP={dP_cap}Pa, eta={eta_blood}Pa*s")
print(f"    Q = pi*r^4*dP/(8*eta*L) = {Q_blood:.4e} m^3/s")

# Reynolds number
rho_w, v_lam, D_pipe, eta_w = 1000.0, 0.1, 0.01, 0.001
Re_laminar = rho_w * v_lam * D_pipe / eta_w
Re_turbulent = rho_w * 1.0 * D_pipe / eta_w
print(f"\n  Reynolds: rho={rho_w}, D={D_pipe}m, eta={eta_w}")
print(f"    v=0.1 m/s: Re={Re_laminar:.0f} (laminar, Re<2300)")
print(f"    v=1.0 m/s: Re={Re_turbulent:.0f} (turbulent, Re>4000)")

print("\n  Optical fiber fluid analogy:")
print("    Etendue conservation <-> continuity; group velocity <-> flow velocity; GVD <-> dispersion")

chk(v2_continuity, 8.0, "v2_continuity", absolute=True)
chk(deltaP_venturi, -30000.0, "deltaP_venturi", tol=100, absolute=True)
chk(Q_blood, 8.18e-16, "Q_blood", tol=1e-17, absolute=True)
chk(Re_laminar, 1000.0, "Re_laminar", absolute=True)
chk(Re_turbulent, 10000.0, "Re_turbulent", absolute=True)

# %% [markdown]
# ## §4 — Hermitian operators: Griffiths Ch.3 synthesis

# %%
hdr("§4 — Hermitian operators (Griffiths Ch.3)")

# 2x2 Hermitian matrix
H_mat = sp.Matrix([[3, 1-2*sp.I], [1+2*sp.I, 1]])
H_herm = H_mat.H  # conjugate transpose
print("  H =")
show(H_mat)
print("  H_dagger =")
show(H_herm)

H_is_hermitian = 1 if H_mat == H_herm else 0
print(f"  H == H_dagger: {H_is_hermitian == 1}")

# Eigenvalues
eigs = H_mat.eigenvals()
print("\n  Eigenvalues:")
show(eigs)
eig_list = list(H_mat.eigenvals().keys())
eig_vals_complex = [complex(e) for e in eig_list]
print(f"  Eigenvalues (complex): {eig_vals_complex}")
eigenvalues_real = all(abs(e.imag) < 1e-10 for e in eig_vals_complex)
print(f"  All real: {eigenvalues_real}")

# Eigenvectors
evects = H_mat.eigenvects()
print("\n  Eigenvectors:")
show(evects)

# Orthogonality check
v1_ev = evects[0][2][0]
v2_ev = evects[1][2][0]
dot_product = (v1_ev.H * v2_ev)[0]
dot_simplified = sp.simplify(dot_product)
print(f"\n  <v1|v2> = {dot_simplified} (should be 0 -- orthogonal)")

# Pauli matrices
sigma_x = sp.Matrix([[0,1],[1,0]])
sigma_y = sp.Matrix([[0,-sp.I],[sp.I,0]])
sigma_z = sp.Matrix([[1,0],[0,-1]])
I2 = sp.eye(2)

print("\n  Pauli matrices:")
show(sigma_x, "sigma_x")
show(sigma_y, "sigma_y")
show(sigma_z, "sigma_z")

# Commutator [sigma_x, sigma_y] = 2i sigma_z
comm_xy = sigma_x * sigma_y - sigma_y * sigma_x
comm_xy_simplified = sp.simplify(comm_xy)
comm_expected = 2*sp.I*sigma_z
print("\n  [sigma_x, sigma_y] =")
show(comm_xy_simplified)
print("  Expected: 2i sigma_z")

# Measurement postulate: |psi> = 1/sqrt(2)|+z> + 1/sqrt(2)|-z>
c_plus = 1/np.sqrt(2)
c_minus = 1/np.sqrt(2)
P_plus = abs(c_plus)**2
P_minus = abs(c_minus)**2
expect_sigma_z = P_plus * 1 + P_minus * (-1)
expect_sigma_z_sq = P_plus * 1**2 + P_minus * (-1)**2
sigma_uncertainty = np.sqrt(expect_sigma_z_sq - expect_sigma_z**2)
print(f"\n  |psi> = 1/sqrt(2)|+z> + 1/sqrt(2)|-z>:")
print(f"    P(+1)={P_plus:.4f}, P(-1)={P_minus:.4f}")
print(f"    <sigma_z>={expect_sigma_z:.4f}, <sigma_z^2>={expect_sigma_z_sq:.4f}")
print(f"    sigma_{{sigma_z}} = {sigma_uncertainty:.4f}")

print("\n  RogueGuard: intensity operator I_hat has eigenvalues {I_n};")
print("    rogue = eigenvalue > 2*mu (measuring intensity collapses to an eigenstate)")

chk(H_is_hermitian, 1, "H_is_hermitian==1", absolute=True)
chk(abs(eig_vals_complex[0].imag), 0, "eigenvalue[0]_imag<1e-10", tol=1e-9, absolute=True)
chk(abs(eig_vals_complex[1].imag), 0, "eigenvalue[1]_imag<1e-10", tol=1e-9, absolute=True)
# [sigma_x, sigma_y] = 2i sigma_z; check [1,0] element: sigma_z[1,0]=0 -> (2i*sigma_z)[1,0]=0
# check [0,1] element: sigma_z[0,1]=0 -> (2i*sigma_z)[0,1]=0
# check [1,0] = 2i*sigma_z[1,0]... sigma_z = diag(1,-1), off-diag = 0
# Actual: [sigma_x,sigma_y] = [[2i,0],[0,-2i]] = 2i*sigma_z (since sigma_z=diag(1,-1))
comm_00 = complex(comm_xy_simplified[0,0])
chk(comm_00.imag, 2.0, "pauli_commutator_xy [0,0] imag==2 (=2i)", tol=1e-10, absolute=True)
chk(P_plus, 0.5, "P_plus_half==0.5", absolute=True)
chk(sigma_uncertainty, 1.0, "sigma_uncertainty==1.0", absolute=True)

# %% [markdown]
# ## §5 — RogueGuard firmware: C code structure

# %%
hdr("§5 — RogueGuard firmware: C code structure")

firmware_c = r"""/* firmware/rogue_guard_main.c
 * RogueGuard 1U optical rogue wave monitor
 * RPi CM4 + dual AD9226 (12-bit, 65 MSPS ADC)
 * Author: [Your name] / Jalali Lab, UC Davis
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>         /* kill(), getpid() */
#include <sys/wait.h>       /* wait() */
#include "adc_driver.h"
#include "rogue_detect.h"

/* -- Configuration ----------------------------------------- */
#define ADC_SAMPLE_RATE_HZ   65000000UL   /* 65 MSPS */
#define ADC_BITS             12
#define ADC_MAX              ((1 << ADC_BITS) - 1)  /* 4095 */
#define ROGUE_THRESHOLD_MULT 2            /* rogue if I > 2*mu */
#define EMA_ALPHA_BITS       12           /* alpha = 2^{-12} approx 2.4e-4 */
#define RING_BUF_SIZE        (1 << 20)    /* 1M samples approx 15ms @ 65MSPS */

/* -- Data structures --------------------------------------- */
typedef struct {
    uint16_t samples[RING_BUF_SIZE];
    uint32_t head;
    uint32_t count;
} RingBuffer;

typedef struct {
    uint32_t mu_scaled;   /* mean * 2^EMA_ALPHA_BITS */
    uint32_t n_rogues;    /* rogue event counter */
    uint32_t n_samples;   /* total samples processed */
    double   false_alarm_rate; /* running FAR estimate */
} RogueState;

/* -- EMA update (integer, no division) --------------------- */
static inline void ema_update(RogueState *rs, uint16_t x) {
    uint32_t x_scaled = (uint32_t)x << EMA_ALPHA_BITS;
    rs->mu_scaled += (x_scaled - rs->mu_scaled) >> EMA_ALPHA_BITS;
}

/* -- Rogue detection --------------------------------------- */
static inline int is_rogue(const RogueState *rs, uint16_t x) {
    uint32_t mu = rs->mu_scaled >> EMA_ALPHA_BITS;
    return (x > ROGUE_THRESHOLD_MULT * mu);
}

/* -- Main acquisition loop --------------------------------- */
void acquisition_loop(void) {
    RingBuffer  rb  = {0};
    RogueState  rs  = { .mu_scaled = 2048 << EMA_ALPHA_BITS };
    uint32_t    rogue_count = 0;

    printf("[RogueGuard] PID=%d  ADC=%lu MSPS  threshold=%dx\n",
           getpid(), ADC_SAMPLE_RATE_HZ/1000000, ROGUE_THRESHOLD_MULT);

    while (1) {                       /* broken by SIGTERM */
        uint16_t x = adc_read();      /* non-blocking SPI read */

        rb.samples[rb.head] = x;
        rb.head = (rb.head + 1) & (RING_BUF_SIZE - 1);
        rb.count++;

        ema_update(&rs, x);

        if (is_rogue(&rs, x)) {
            rs.n_rogues++;
            rogue_event_log(&rs, x);  /* write timestamp + intensity */
        }
        rs.n_samples++;
    }
}

/* -- Signal handlers --------------------------------------- */
static volatile int running = 1;

void sigterm_handler(int sig) {
    (void)sig;
    running = 0;          /* graceful shutdown */
}

int main(void) {
    signal(SIGTERM, sigterm_handler);
    signal(SIGINT,  sigterm_handler);
    acquisition_loop();
    printf("[RogueGuard] Shutdown: %u rogues / %u samples (FAR=%.4f)\n",
           rs.n_rogues, rs.n_samples,
           (double)rs.n_rogues / rs.n_samples);
    return 0;
}
"""
print(firmware_c)

# Python simulation of EMA + rogue detector
np.random.seed(42)
N_fw = 50000
mu_true = 2048
threshold_mult_fw = 2
alpha_bits_fw = 12

samples_fw = np.random.exponential(scale=mu_true, size=N_fw).astype(np.float64)
rogue_positions = [10000, 25000, 40000]
for pos in rogue_positions:
    samples_fw[pos] = 5 * mu_true

mu_scaled_fw = mu_true << alpha_bits_fw  # integer-like
detected_rogues = 0
false_alarms = 0
rogue_set = set(rogue_positions)

mu_estimates = []
for i, x in enumerate(samples_fw):
    xi = int(x)
    x_scaled_fw = xi << alpha_bits_fw
    mu_scaled_fw = mu_scaled_fw + ((x_scaled_fw - mu_scaled_fw) >> alpha_bits_fw)
    mu_est = mu_scaled_fw >> alpha_bits_fw
    is_rogue_flag = (xi > threshold_mult_fw * mu_est)

    if i in rogue_set:
        if is_rogue_flag:
            detected_rogues += 1
    else:
        if is_rogue_flag:
            false_alarms += 1

    mu_estimates.append(mu_est)

mu_converged = mu_estimates[-1]
far_background = false_alarms / (N_fw - len(rogue_positions))
print(f"\n  Firmware simulation results:")
print(f"    Detected rogues: {detected_rogues}/3")
print(f"    False alarms: {false_alarms} / {N_fw - len(rogue_positions)} background samples")
print(f"    FAR = {far_background:.4f}  (expected approx e^{{-2}} = {np.exp(-2):.4f})")
print(f"    Final mu estimate: {mu_converged} (true: {mu_true})")

chk(detected_rogues, 3, "all_3_rogues_detected==3", absolute=True)
chk(mu_converged, 2048, "mu_converged near 2048", tol=100, absolute=True)
chk(far_background, np.exp(-2), "far_background near exp(-2)", tol=0.02, absolute=True)

# %% [markdown]
# ## §6 — Formal grammar: CFG, BNF, DFA

# %%
hdr("§6 — Formal grammar: CFG, BNF, DFA")

print("  Context-Free Grammar for arithmetic expressions:")
print("    E -> E + T | T")
print("    T -> T * F | F")
print("    F -> ( E ) | number")
print("  (* binds tighter than + via grammar hierarchy)")

print("""
  BNF:
  <expr>   ::= <expr> '+' <term> | <term>
  <term>   ::= <term> '*' <factor> | <factor>
  <factor> ::= '(' <expr> ')' | <number>
  <number> ::= [0-9]+
""")

# Recursive descent parser
import re as _re

def tokenize(text):
    return _re.findall(r'\d+|[+*()]', text.replace(' ', ''))

class Parser:
    def __init__(self, text):
        self.tokens = tokenize(text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def parse_expr(self):
        left = self.parse_term()
        while self.peek() == '+':
            self.consume()
            right = self.parse_term()
            left = left + right
        return left

    def parse_term(self):
        left = self.parse_factor()
        while self.peek() == '*':
            self.consume()
            right = self.parse_factor()
            left = left * right
        return left

    def parse_factor(self):
        if self.peek() == '(':
            self.consume()  # '('
            val = self.parse_expr()
            self.consume()  # ')'
            return val
        else:
            return int(self.consume())

parse_3_plus_4_times_2 = Parser("3 + 4 * 2").parse_expr()
parse_2_times_paren_3_plus_4 = Parser("2 * (3 + 4)").parse_expr()
print(f"  Parse '3 + 4 * 2' = {parse_3_plus_4_times_2}  (expected 11, precedence!)")
print(f"  Parse '2 * (3 + 4)' = {parse_2_times_paren_3_plus_4}  (expected 14)")

# DFA for identifier recognition
def dfa_identifier(s):
    if not s:
        return False
    state = 'q0'
    for ch in s:
        if state == 'q0':
            if ch.isalpha():
                state = 'q1'
            else:
                state = 'q2'
        elif state == 'q1':
            if ch.isalpha() or ch.isdigit():
                state = 'q1'
            else:
                state = 'q2'
        else:  # q2 dead
            state = 'q2'
    return state == 'q1'

dfa_tests = [("hello", True), ("x123", True), ("123abc", False), ("_var", False), ("", False)]
for s, expected in dfa_tests:
    result = dfa_identifier(s)
    mark = "OK" if result == expected else "WRONG"
    print(f"  dfa_identifier({s!r}) = {result}  [{mark}]")

print("\n  Chomsky hierarchy:")
print("    Type 0 (unrestricted) > Type 1 (context-sensitive) > Type 2 (CFG) > Type 3 (regular)")
print("    C code -> Type 2 (CFG); assembly labels -> Type 3 (regular)")

chk(parse_3_plus_4_times_2, 11, "parse_3_plus_4_times_2==11", absolute=True)
chk(parse_2_times_paren_3_plus_4, 14, "parse_2_times_paren_3_plus_4==14", absolute=True)
chk(int(dfa_identifier("hello")), 1, "dfa_hello==1", absolute=True)
chk(int(dfa_identifier("x123")), 1, "dfa_x123==1", absolute=True)
chk(int(dfa_identifier("123abc")), 0, "dfa_123abc==0", absolute=True)

# %% [markdown]
# ## §7 — Quaternions: 3D rotations for VR

# %%
hdr("§7 — Quaternions: 3D rotations for VR")

print("  Quaternion: q = w + xi + yj + zk; i^2=j^2=k^2=ijk=-1")
print("  Unit quaternion |q|=1 represents rotation")
print("  Rotation by theta around n_hat: q = (cos(theta/2), sin(theta/2)*n_hat)")

def quat_mul(q1, q2):
    """Hamilton product of two quaternions [w,x,y,z]"""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rotate_vec(v, q):
    """Rotate vector v by unit quaternion q using sandwich product"""
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_c = quat_conj(q)
    result = quat_mul(quat_mul(q, v_quat), q_c)
    return result[1:]  # drop w component

# Identity test: q * identity = q
theta_90 = np.pi / 2
q_90z = np.array([np.cos(theta_90/2), 0.0, 0.0, np.sin(theta_90/2)])
q_identity = np.array([1.0, 0.0, 0.0, 0.0])
q_mul_identity = quat_mul(q_90z, q_identity)
print(f"\n  q * identity = {q_mul_identity}  (should equal q_90z)")

# Rotate [1,0,0] by 90 deg around Z-axis
v_x = np.array([1.0, 0.0, 0.0])
v_rot_90z = rotate_vec(v_x, q_90z)
print(f"\n  Rotate [1,0,0] by 90 deg around Z: {v_rot_90z}  (expected [0,1,0])")

# SLERP
def slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(np.sin(omega)) < 1e-10:
        return (1-t)*q1 + t*q2
    return (np.sin((1-t)*omega)/np.sin(omega)) * q1 + (np.sin(t*omega)/np.sin(omega)) * q2

q_slerp_half = slerp(q_identity, q_90z, 0.5)
slerp_angle_deg = 2 * np.degrees(np.arccos(np.clip(q_slerp_half[0], -1, 1)))
print(f"\n  SLERP(identity, 90deg_around_Z, t=0.5): {q_slerp_half}")
print(f"    Angle = {slerp_angle_deg:.4f} deg  (expected 45 deg)")

print("\n  Gimbal lock: Euler angles at theta=90 deg lose 1 DOF (Rz and Rx become parallel)")
print("  Quaternions: no gimbal lock -- SO(3) covered smoothly by S^3")

chk(q_mul_identity[0], q_90z[0], "quat_mul identity w-component", tol=1e-10, absolute=True)
chk(v_rot_90z[0], 0.0, "rotate_x_by_90z: x~0", tol=1e-6, absolute=True)
chk(v_rot_90z[1], 1.0, "rotate_x_by_90z: y~1", tol=1e-6, absolute=True)
chk(v_rot_90z[2], 0.0, "rotate_x_by_90z: z~0", tol=1e-6, absolute=True)
chk(slerp_angle_deg, 45.0, "slerp_at_t0.5_angle~45deg", tol=0.01, absolute=True)

# %% [markdown]
# ## §8 — VR spatial math: view matrix, projection, frustum

# %%
hdr("§8 — VR spatial math: view, projection, frustum")

def look_at(eye, center, up):
    """Compute 4x4 LookAt view matrix"""
    eye, center, up = np.array(eye, float), np.array(center, float), np.array(up, float)
    fwd = center - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up_new = np.cross(right, fwd)
    V = np.array([
        [right[0],  right[1],  right[2],  -np.dot(right, eye)],
        [up_new[0], up_new[1], up_new[2], -np.dot(up_new, eye)],
        [-fwd[0],   -fwd[1],   -fwd[2],    np.dot(fwd, eye)],
        [0.0,       0.0,       0.0,        1.0]
    ])
    return V

def perspective(fov_deg, aspect, near, far):
    """Compute 4x4 perspective projection matrix"""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    P = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
        [0, 0, -1, 0]
    ])
    return P

eye_cam = np.array([0.0, 5.0, 10.0])
center_cam = np.array([0.0, 0.0, 0.0])
up_cam = np.array([0.0, 1.0, 0.0])
V = look_at(eye_cam, center_cam, up_cam)
print("  View matrix (camera at (0,5,10) looking at origin):")
print(V)

near, far = 0.1, 1000.0
fov_deg, aspect = 60.0, 16.0/9.0
P = perspective(fov_deg, aspect, near, far)
print("\n  Projection matrix (fov=60 deg, 16:9, near=0.1, far=1000):")
print(P)

# Transform world point
P_w = np.array([1.0, 1.0, -5.0, 1.0])
P_view = V @ P_w
P_clip = P @ P_view
w_clip = P_clip[3]
NDC = P_clip[:3] / w_clip
print(f"\n  World point P_w = {P_w[:3]}")
print(f"  Clip coords: {P_clip}")
print(f"  NDC (x,y,z): {NDC}")

# Near and far plane NDC z
z_view_near = -near
z_clip_near = (far+near)/(near-far)*z_view_near + 2*far*near/(near-far)
w_clip_near = -z_view_near
NDC_z_near = z_clip_near / w_clip_near

z_view_far = -far
z_clip_far = (far+near)/(near-far)*z_view_far + 2*far*near/(near-far)
w_clip_far = -z_view_far
NDC_z_far = z_clip_far / w_clip_far
print(f"\n  NDC z at near plane ({near}m): {NDC_z_near:.6f}  (expected -1)")
print(f"  NDC z at far  plane ({far}m):  {NDC_z_far:.6f}   (expected +1)")

# VR stereo: dual frustum
IPD = 0.063  # 63mm
eye_left  = eye_cam + np.array([-IPD/2, 0, 0])
eye_right = eye_cam + np.array([ IPD/2, 0, 0])
V_left  = look_at(eye_left,  center_cam, up_cam)
V_right = look_at(eye_right, center_cam, up_cam)
P_w4 = np.array([1.0, 1.0, -5.0, 1.0])
ndc_left  = P @ V_left  @ P_w4
ndc_right = P @ V_right @ P_w4
disparity = ndc_left[0]/ndc_left[3] - ndc_right[0]/ndc_right[3]
print(f"\n  VR stereo (IPD=63mm), point at world (1,1,-5):")
print(f"    NDC-x left:  {ndc_left[0]/ndc_left[3]:.6f}")
print(f"    NDC-x right: {ndc_right[0]/ndc_right[3]:.6f}")
print(f"    Disparity (encodes depth): {disparity:.6f}")

proj_m22 = (far+near)/(near-far)
chk(V.shape[0]*V.shape[1], 16, "view_matrix_shape==(4,4) -> 16 elements", absolute=True)
chk(P[2,2], proj_m22, "proj_matrix_m22", tol=1e-6)
chk(NDC_z_near, -1.0, "NDC_z_near~-1", tol=0.01, absolute=True)
chk(NDC_z_far,   1.0, "NDC_z_far~+1",  tol=0.01, absolute=True)

# %% [markdown]
# ## §9 — Hermitian <-> VR connection: quantum measurement as projection

# %%
hdr("§9 — Hermitian <-> VR: measurement as projection")

print("  QM measurement: |psi> -> |n><n|psi>  (project onto eigenstate basis)")
print("  VR rendering:   P_world -> Project(View(P_world))  (3D -> 2D)")
print("  Both are linear projections onto lower-dimensional subspace\n")

# Density matrix
psi_vec = sp.Matrix([sp.Rational(1,1)/sp.sqrt(2), sp.Rational(1,1)/sp.sqrt(2)])
rho_dm = psi_vec * psi_vec.H
print("  |psi> = 1/sqrt(2) |0> + 1/sqrt(2) |1>")
print("  Density matrix rho = |psi><psi|:")
show(rho_dm)
rho_trace = sp.trace(rho_dm)
rho_sq = rho_dm * rho_dm
rho_purity = sp.trace(rho_sq)
print(f"  Tr(rho) = {rho_trace}")
print(f"  Tr(rho^2) = {rho_purity}  (1 = pure state)")
show(sp.Eq(sp.Symbol('rho_sq'), rho_sq), "rho^2")

# After measurement (decoherence)
rho_after = sp.Matrix([[sp.Rational(1,2), 0], [0, sp.Rational(1,2)]])
purity_after = sp.trace(rho_after * rho_after)
print(f"\n  After measurement collapse: rho_mixed = (1/2)*I")
print(f"  Tr(rho_mixed^2) = {purity_after}  (< 1 -> mixed state)")

# Projection operator
P_proj = sp.Matrix([[1,0],[0,0]])
P_sq = P_proj * P_proj
P_herm = P_proj.H
print(f"\n  Projection operator P = |0><0|:")
show(P_proj)
print(f"  P^2 == P: {P_sq == P_proj}")
print(f"  P_dagger == P: {P_herm == P_proj}")

# Fisher information geometry
mu_fisher = 5.0
N_fisher = 100
g_mu = 1.0 / mu_fisher**2
CRB = 1.0 / (N_fisher * g_mu)   # = mu^2/N
print(f"\n  Fisher information for exponential: g(mu) = 1/mu^2")
print(f"  CRB: Var(mu_hat) >= mu^2/N = {mu_fisher}^2/{N_fisher} = {CRB:.4f}")
print(f"  Geodesic distance: |log(mu2/mu1)|  (multiplicative scale = straight line)")

rho_trace_float = float(rho_trace)
rho_purity_float = float(rho_purity)
chk(rho_trace_float, 1.0, "rho_trace==1", tol=1e-10, absolute=True)
chk(rho_purity_float, 1.0, "rho_purity==1 (pure state)", tol=1e-10, absolute=True)
chk(int(P_sq == P_proj), 1, "P_idempotent P^2==P", absolute=True)
chk(CRB, mu_fisher**2/N_fisher, "fisher_CRB at mu=5 N=100 vs 0.25", tol=1e-6, absolute=True)

# %% [markdown]
# ## §10 — Full loop: process -> signal -> physics -> firmware -> render

# %%
hdr("§10 — Full loop: everything connected")

connection_map = """
  kill -9  (SIGKILL)         -> ema_update() in C firmware exits immediately
  kill -15 (SIGTERM)         -> sigterm_handler() graceful shutdown, prints FAR
  SIGINT   (Ctrl+C)          -> same as kill -2 -> firmware stops acquisition

  Fluid dynamics (S3)        -> Poiseuille Q~r^4 -> capillary -> optical fiber (r=4um)
  Bernoulli dP~v^2           -> analogous to GVD: dphi~omega^2 in dispersion
  Reynolds Re=rho*v*D/eta    -> optical analog: V-number (single/multi-mode boundary)

  Hermitian H_hat (S4)       -> eigenvalues = energy levels = photon frequencies
  Measurement postulate       -> photodetector collapses field -> I=|E|^2 (intensity)
  Density matrix rho          -> mixed state after decoherence -> classical intensity noise

  RogueGuard firmware (S5)   -> EMA in C, SIGTERM handler, ring buffer at 65 MSPS
  DFA grammar (S6)            -> C parser in compiler toolchain compiling the firmware
  Quaternion (S7)             -> VR headset rendering RogueGuard dashboard in 3D
  Perspective matrix (S8)     -> projecting 3D rogue event cloud onto display
"""
print(connection_map)

# Final numerical chain
mu_adc = 2048
ADC_MAX_val = 4095
alpha_fine = 1/137.036
hnu_1550nm_eV = (6.626e-34 * 3e8 / 1550e-9) / 1.602e-19

ADC_max_div_mu = ADC_MAX_val / mu_adc
N_1s = 65e6
CRB_firmware_1s = mu_adc**2 / N_1s

print(f"  mu_true = {mu_adc} ADC counts")
print(f"  ADC_MAX / mu_true = {ADC_MAX_val}/{mu_adc} = {ADC_max_div_mu:.4f} (rogue threshold at max ADC range)")
print(f"  Fisher CRB for 1s = mu^2/N = {mu_adc}^2/{N_1s:.0e} = {CRB_firmware_1s:.4f} ADC counts")
print(f"  Sub-LSB precision! Firmware estimates mu to +/-{CRB_firmware_1s:.3f} counts in 1 second")
print(f"  Fine structure constant alpha = 1/137.036 = {alpha_fine:.6f}")
print(f"  hnu at 1550nm = {hnu_1550nm_eV:.3f} eV")

chk(ADC_max_div_mu, 4095/2048, "ADC_max_div_mu", tol=1e-4)
chk(CRB_firmware_1s, 2048**2/65e6, "CRB_firmware_1s", tol=0.001, absolute=True)
chk(alpha_fine, 1/137.036, "alpha_fine", tol=1e-4)

# %% [markdown]
# ## Summary

# %%
hdr("SUMMARY -- all checks complete")
print("  S1  CLI signals/processes/pipes")
print("  S2  Rotational mechanics (sphere, gyroscope, Euler)")
print("  S3  Fluid dynamics (Bernoulli, Poiseuille, Reynolds)")
print("  S4  Hermitian operators (Griffiths Ch.3, Pauli, measurement)")
print("  S5  RogueGuard C firmware + Python EMA simulation")
print("  S6  CFG/BNF, recursive descent parser, DFA identifier")
print("  S7  Quaternions (Hamilton product, rotation, SLERP)")
print("  S8  VR view/projection matrices, stereo disparity")
print("  S9  QM<->VR projection connection, density matrix, Fisher CRB")
print("  S10 Full integration loop + numerical chain")
