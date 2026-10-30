// Discrete-event simulation (DES) engine and an M/M/c queue model. C++17, header-only.
//
// A DES keeps a *future event list* (a min-priority queue ordered by time). The clock never
// ticks; it jumps straight to the next event, which may schedule more events. Here the events
// are ARRIVAL (a job shows up) and DEPARTURE (a server finishes a job). With Poisson arrivals
// (rate lambda), exponential service (rate mu) and c servers, this is the M/M/c queue, whose
// exact steady-state answers (Erlang C) let us check the simulator.
//
// Example framing used in the notebook: collapse reports in a disaster arrive at random, and
// c rescue teams respond first-come first-served.
#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <queue>
#include <random>
#include <vector>

namespace evsim {

enum EventType { ARRIVAL = 0, DEPARTURE = 1 };

struct Event {
    double t;
    EventType type;
    long id;  // job number, in arrival order
};

// Earliest time first; ties broken by job id so runs are reproducible.
struct Later {
    bool operator()(const Event& a, const Event& b) const {
        return a.t > b.t || (a.t == b.t && a.id > b.id);
    }
};

class EventList {
public:
    void schedule(const Event& e) { q_.push(e); }
    bool empty() const { return q_.empty(); }
    Event pop() { Event e = q_.top(); q_.pop(); return e; }
private:
    std::priority_queue<Event, std::vector<Event>, Later> q_;
};

struct Result {
    long measured = 0;        // jobs after warm-up whose departure was recorded
    double mean_wait = 0;     // Wq: time in line before service starts
    double mean_system = 0;   // W: time from arrival to departure
    double p_wait = 0;        // fraction of jobs that had to wait at all
    double mean_in_system = 0;  // L: time-average number of jobs present
    double utilization = 0;     // busy servers / c, time-averaged
    double arrival_rate = 0;    // observed arrivals per unit time in the window
    long events = 0;
};

// Simulate warmup + n jobs. Statistics use jobs with id >= warmup (time averages use the window
// from job `warmup`'s arrival to the last job's arrival, so the final drain is excluded).
// If trace != nullptr, the first trace_max events are written as CSV: t,type,id,busy,queue.
inline Result run_mmc(double lam, double mu, int c, long n, long warmup, std::uint64_t seed,
                      std::FILE* trace = nullptr, long trace_max = 0) {
    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> interarrival(lam), service(mu);
    const long total = warmup + n;

    EventList fel;
    std::deque<long> line;
    std::vector<double> arrive, start;
    arrive.reserve(total);
    start.reserve(total);

    int busy = 0;
    long in_system = 0, logged = 0, window_arrivals = 0, waited = 0;
    double last = 0, area_sys = 0, area_busy = 0, t0 = 0, t1 = 0, sum_wait = 0, sum_sys = 0;
    bool measuring = false;
    Result res;

    fel.schedule({interarrival(rng), ARRIVAL, 0});
    while (!fel.empty()) {
        const Event e = fel.pop();
        const double now = e.t;
        if (measuring) {
            area_sys += in_system * (now - last);
            area_busy += busy * (now - last);
        }
        last = now;
        ++res.events;

        if (e.type == ARRIVAL) {
            if (e.id == warmup) { measuring = true; t0 = now; }
            if (measuring) ++window_arrivals;
            arrive.push_back(now);
            start.push_back(-1.0);
            ++in_system;
            if (busy < c) {
                ++busy;
                start[e.id] = now;
                fel.schedule({now + service(rng), DEPARTURE, e.id});
            } else {
                line.push_back(e.id);
            }
            if (e.id + 1 < total) {
                fel.schedule({now + interarrival(rng), ARRIVAL, e.id + 1});
            } else {
                measuring = false;
                t1 = now;
            }
        } else {
            --in_system;
            if (e.id >= warmup) {
                const double w = start[e.id] - arrive[e.id];
                sum_wait += w;
                sum_sys += now - arrive[e.id];
                if (w > 0) ++waited;
                ++res.measured;
            }
            if (!line.empty()) {
                const long next = line.front();
                line.pop_front();
                start[next] = now;
                fel.schedule({now + service(rng), DEPARTURE, next});
            } else {
                --busy;
            }
        }
        if (trace && logged < trace_max) {
            std::fprintf(trace, "%.9f,%d,%ld,%d,%zu\n", now, static_cast<int>(e.type), e.id, busy, line.size());
            ++logged;
        }
    }

    const double span = t1 - t0;
    res.mean_wait = sum_wait / res.measured;
    res.mean_system = sum_sys / res.measured;
    res.p_wait = static_cast<double>(waited) / res.measured;
    res.mean_in_system = span > 0 ? area_sys / span : 0;
    res.utilization = span > 0 ? area_busy / (c * span) : 0;
    res.arrival_rate = span > 0 ? (window_arrivals - 1) / span : 0;
    return res;
}

}  // namespace evsim
