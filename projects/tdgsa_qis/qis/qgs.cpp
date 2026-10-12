// qgs.cpp -- Quantum Gerchberg-Saxton state recovery with shot noise (C++17 + POSIX threads)
//
// A 6-qubit state |psi> (chirped Gaussian wave packet) is measured in two bases:
//   basis 1: computational basis after D(a1)   (a1 = 0, no dispersion)
//   basis 2: computational basis after D(a2)   (a2 = 4)
// where D(a) = QFT^-1 . diag(exp(-i a k^2)) . QFT is the discrete dispersion /
// free-particle propagator. Each basis is sampled with a finite number of shots,
// GS recovers the complex amplitudes, and the fidelity |<psi_rec|psi>|^2 is recorded.
// Many independent trials run in parallel on POSIX threads.
//
// Build:  g++ -std=c++17 -O2 -Wall -Wextra -pthread qgs.cpp -o qgs_cpu.exe
// Run:    qgs_cpu.exe <trials> <threads> <out.csv>
//
// Colin Casey, 2026. Extends G. Morozowsky and C. Casey, ECE 279AS (UCLA), 2024.
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using cd = std::complex<double>;
constexpr int NQ = 6, DIM = 1 << NQ, GS_ITERS = 300;
constexpr double PI = 3.14159265358979323846, A1 = 0.0, A2 = 4.0;

// In-place radix-2 FFT, unscaled. sign = -1: sum x e^{-i..}; sign = +1: sum x e^{+i..}
static void fft(std::vector<cd> &x, int sign) {
    const int n = static_cast<int>(x.size());
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const cd wl = std::polar(1.0, sign * 2.0 * PI / len);
        for (int i = 0; i < n; i += len) {
            cd w = 1.0;
            for (int k = 0; k < len / 2; k++) {
                cd u = x[i + k], v = x[i + k + len / 2] * w;
                x[i + k] = u + v; x[i + k + len / 2] = u - v; w *= wl;
            }
        }
    }
}

static std::vector<double> kgrid() {
    std::vector<double> k(DIM);
    for (int j = 0; j < DIM; j++) k[j] = 2.0 * PI * (j < DIM / 2 ? j : j - DIM) / DIM;
    return k;
}
static const std::vector<double> K = kgrid();

// D(a)|v> = QFT^-1 diag(e^{-i a k^2}) QFT |v>, matching Qiskit's QFTGate convention
static std::vector<cd> disperse(double a, std::vector<cd> v) {
    const double s = 1.0 / std::sqrt(static_cast<double>(DIM));
    fft(v, +1);
    for (int j = 0; j < DIM; j++) v[j] *= s * std::polar(1.0, -a * K[j] * K[j]);
    fft(v, -1);
    for (auto &z : v) z *= s;
    return v;
}

static std::vector<cd> make_state() {
    std::vector<cd> psi(DIM);
    double norm = 0.0;
    for (int i = 0; i < DIM; i++) {
        double x = i - DIM / 2;
        cd val = std::exp(cd(-x * x / 16.0, 0.15 * x * x / 4.0));
        psi[(i + DIM / 2) % DIM] = val;                 // np.roll(..., DIM/2)
        norm += std::norm(val);
    }
    for (auto &z : psi) z /= std::sqrt(norm);
    return psi;
}

static std::vector<double> probabilities(const std::vector<cd> &amp) {
    std::vector<double> p(DIM);
    for (int j = 0; j < DIM; j++) p[j] = std::norm(amp[j]);
    return p;
}

static double gs_fidelity(const std::vector<double> &P1, const std::vector<double> &P2, const std::vector<cd> &psi) {
    std::vector<double> m1(DIM), m2(DIM);
    for (int j = 0; j < DIM; j++) { m1[j] = std::sqrt(P1[j]); m2[j] = std::sqrt(P2[j]); }
    std::vector<cd> f(m1.begin(), m1.end());
    for (int it = 0; it < GS_ITERS; it++) {
        std::vector<cd> g = disperse(A2 - A1, f);
        for (int j = 0; j < DIM; j++) g[j] = std::polar(m2[j], std::arg(g[j]));
        f = disperse(A1 - A2, g);
        for (int j = 0; j < DIM; j++) f[j] = std::polar(m1[j], std::arg(f[j]));
    }
    double n2 = 0.0; cd ov = 0.0;
    for (int j = 0; j < DIM; j++) { n2 += std::norm(f[j]); ov += std::conj(f[j]) * psi[j]; }
    return std::norm(ov) / n2;
}

static std::vector<double> sample(const std::vector<double> &P, long shots, std::mt19937_64 &rng) {
    std::discrete_distribution<int> dist(P.begin(), P.end());
    std::vector<double> c(DIM, 0.0);
    for (long s = 0; s < shots; s++) c[dist(rng)] += 1.0;
    for (auto &x : c) x /= static_cast<double>(shots);
    return c;
}

struct Job {
    long shots;
    int first, last;
    const std::vector<double> *P1, *P2;
    const std::vector<cd> *psi;
    double *out;
};

static void *worker(void *arg) {
    Job *job = static_cast<Job *>(arg);
    for (int tr = job->first; tr < job->last; tr++) {
        std::mt19937_64 rng(1000003ULL * static_cast<unsigned long long>(job->shots) + static_cast<unsigned long long>(tr));
        std::vector<double> p1 = sample(*job->P1, job->shots, rng);
        std::vector<double> p2 = sample(*job->P2, job->shots, rng);
        job->out[tr] = gs_fidelity(p1, p2, *job->psi);
    }
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc < 4) { std::cerr << "usage: " << argv[0] << " <trials> <threads> <out.csv>\n"; return 1; }
    const int trials = std::atoi(argv[1]), nthreads = std::atoi(argv[2]);
    const std::string csv = argv[3];

    const std::vector<cd> psi = make_state();
    const std::vector<double> P1 = probabilities(disperse(A1, psi)), P2 = probabilities(disperse(A2, psi));

    std::ofstream out(csv);
    out << "backend,shots,trial,fidelity\n" << std::setprecision(10);
    std::ofstream pr(csv + ".probs.csv");                     // exact Born probabilities for cross-checking
    pr << "j,P1,P2\n" << std::setprecision(15);
    for (int j = 0; j < DIM; j++) pr << j << ',' << P1[j] << ',' << P2[j] << '\n';

    const double exact = gs_fidelity(P1, P2, psi);
    out << "cpu,0,0," << exact << '\n';
    std::cout << "backend=C++/POSIX threads=" << nthreads << " trials=" << trials << " qubits=" << NQ << '\n';
    std::cout << "exact probabilities -> fidelity " << std::setprecision(8) << exact << "\n\n";
    std::cout << std::setw(8) << "shots" << std::setw(14) << "mean F" << std::setw(12) << "time (s)" << '\n';

    for (long shots : {1L, 10L, 100L, 1000L, 10000L, 100000L}) {
        std::vector<double> fid(trials);
        std::vector<pthread_t> th(nthreads);
        std::vector<Job> jobs(nthreads);
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < nthreads; t++) {
            jobs[t] = {shots, trials * t / nthreads, trials * (t + 1) / nthreads, &P1, &P2, &psi, fid.data()};
            pthread_create(&th[t], nullptr, worker, &jobs[t]);
        }
        for (auto &h : th) pthread_join(h, nullptr);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double mean = 0.0;
        for (int tr = 0; tr < trials; tr++) { mean += fid[tr]; out << "cpu," << shots << ',' << tr << ',' << fid[tr] << '\n'; }
        std::cout << std::setw(8) << shots << std::setw(14) << std::setprecision(6) << mean / trials
                  << std::setw(12) << std::setprecision(3) << secs << '\n';
    }
    return 0;
}
