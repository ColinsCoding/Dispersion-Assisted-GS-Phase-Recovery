// Command-line front end for the M/M/c event simulator.
//
//   eventsim lambda mu c n warmup seed [trace.csv trace_max]
//
// Prints key=value lines. Exit code 2 on bad arguments.
// Build: g++ -O2 -std=c++17 -Wall -Wextra -static eventsim_cli.cpp -o eventsim
#include <cstdio>
#include <cstdlib>

#include "eventsim.hpp"

int main(int argc, char** argv) {
    if (argc != 7 && argc != 9) {
        std::fprintf(stderr, "usage: %s lambda mu c n warmup seed [trace.csv trace_max]\n", argv[0]);
        return 2;
    }
    const double lam = std::atof(argv[1]), mu = std::atof(argv[2]);
    const int c = std::atoi(argv[3]);
    const long n = std::atol(argv[4]), warmup = std::atol(argv[5]);
    const unsigned long long seed = std::strtoull(argv[6], nullptr, 10);
    if (lam <= 0 || mu <= 0 || c < 1 || n < 1 || warmup < 0) {
        std::fprintf(stderr, "need lambda > 0, mu > 0, c >= 1, n >= 1, warmup >= 0\n");
        return 2;
    }
    if (lam >= c * mu) {
        std::fprintf(stderr, "unstable: lambda >= c*mu, the line grows without bound\n");
        return 2;
    }
    std::FILE* trace = nullptr;
    long trace_max = 0;
    if (argc == 9) {
        trace = std::fopen(argv[7], "w");
        if (!trace) { std::fprintf(stderr, "cannot open %s\n", argv[7]); return 2; }
        trace_max = std::atol(argv[8]);
    }
    const evsim::Result r = evsim::run_mmc(lam, mu, c, n, warmup, seed, trace, trace_max);
    if (trace) std::fclose(trace);
    std::printf("measured=%ld\nevents=%ld\nmean_wait=%.12g\nmean_system=%.12g\np_wait=%.12g\n"
                "mean_in_system=%.12g\nutilization=%.12g\narrival_rate=%.12g\n",
                r.measured, r.events, r.mean_wait, r.mean_system, r.p_wait,
                r.mean_in_system, r.utilization, r.arrival_rate);
    return 0;
}
