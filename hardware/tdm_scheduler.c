/* tdm_scheduler.c -- round-robin time-division-multiplexing (TDM) scheduler,
 * the C "golden model" for tdm_scheduler.v.
 *
 * THE PROBLEM: several optical channels (e.g. the depth planes in
 * dgs/steam_3d_depth_encoding.py's proposed time-multiplexed 3D STEAM
 * camera, or several wavelength channels in a WDM link) need to share ONE
 * piece of hardware -- one photodetector, one ADC, one digitizer -- because
 * that hardware is expensive/fast and the channels aren't. TDM is the
 * answer: round-robin through the channels, one per hardware clock cycle
 * (or, more generally, one per fixed-width SLOT of several cycles).
 *
 * tdm_active_channel() is the primitive both this file and tdm_scheduler.v
 * implement (the RTL is exactly a free-running mod-N counter -- see that
 * file's header comment); tdm_scheduler.v's testbench hardcodes this
 * function's output for n_channels=4 and cross-checks the RTL against it
 * cycle-by-cycle. channel_at_cycle() generalizes it to multi-cycle SLOTS,
 * which is what makes the frame_period/slot_duration/effective_rate
 * cycles-per-second numbers below meaningful.
 *
 * Build & run:  gcc -O2 -o tdm_scheduler_c tdm_scheduler.c -lm  &&  ./tdm_scheduler_c
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* THE RTL PRIMITIVE: which channel owns hardware clock cycle `cycle_index`,
 * one cycle per channel, wrapping every n_channels cycles. This is exactly
 * tdm_scheduler.v's free-running mod-N counter (channel_sel). */
int tdm_active_channel(long cycle_index, int n_channels)
{
    if (n_channels <= 0) {
        fprintf(stderr, "tdm_active_channel: n_channels must be > 0, got %d\n", n_channels);
        exit(1);
    }
    return (int)(cycle_index % n_channels);
}

typedef struct {
    double f_clk_hz;        /* hardware clock rate, cycles/sec               */
    int    n_channels;      /* number of time-multiplexed channels           */
    int    cycles_per_slot; /* clock cycles a channel holds before switching */
} tdm_config_t;

/* generalizes tdm_active_channel() to multi-cycle slots: cycles_per_slot=1
 * reduces exactly to tdm_active_channel(cycle_index, n_channels). */
static int channel_at_cycle(tdm_config_t cfg, long cycle_index)
{
    return tdm_active_channel(cycle_index / cfg.cycles_per_slot, cfg.n_channels);
}

static double frame_period_s(tdm_config_t cfg)
{
    return (double)(cfg.n_channels * cfg.cycles_per_slot) / cfg.f_clk_hz;
}

static double slot_duration_s(tdm_config_t cfg)
{
    return (double)cfg.cycles_per_slot / cfg.f_clk_hz;
}

static double effective_channel_rate_hz(tdm_config_t cfg)
{
    return 1.0 / frame_period_s(cfg);
}

static int check(const char *name, int got, int expect)
{
    int ok = got == expect;
    printf("%-46s got=%-6d expect=%-6d  %s\n", name, got, expect, ok ? "OK" : "FAIL");
    return ok;
}

static int check_close(const char *name, double got, double expect, double tol)
{
    int ok = fabs(got - expect) < tol;
    printf("%-46s got=%-.6e expect=%-.6e  %s\n", name, got, expect, ok ? "OK" : "FAIL");
    return ok;
}

int main(void)
{
    int pass = 1;

    /* ---- the RTL cross-check demo: 100 MHz, 4 channels, 1 cycle/slot ----
     * exactly what tdm_scheduler_tb.v drives the Verilog with and hardcodes
     * as its expected sequence. */
    printf("=== RTL cross-check demo: 100 MHz, 4 channels, 1 cycle/slot ===\n");
    const double f_clk_rtl = 100e6;     /* matches tdm_scheduler_tb.v's 10ns clock period */
    const int n_ch_rtl = 4;
    printf("(f_clk = %.1e Hz)\n", f_clk_rtl);
    printf("golden channel sequence (16 cycles): ");
    int golden[16];
    for (int i = 0; i < 16; i++) {
        golden[i] = tdm_active_channel(i, n_ch_rtl);
        printf("%d ", golden[i]);
    }
    printf("\n(tdm_scheduler_tb.v hardcodes this exact sequence and checks the\n"
           " Verilog RTL against it cycle-by-cycle)\n\n");

    static const int expect_golden[16] = {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3};
    for (int i = 0; i < 16; i++)
        pass &= check("golden[i] matches hardcoded RTL testbench expectation",
                       golden[i], expect_golden[i]);

    /* the direct cost of time-sharing, for the RTL's exact configuration */
    tdm_config_t cfg_rtl = { .f_clk_hz = f_clk_rtl, .n_channels = n_ch_rtl, .cycles_per_slot = 1 };
    double frame_period_rtl = frame_period_s(cfg_rtl);
    double eff_rate_rtl = effective_channel_rate_hz(cfg_rtl);
    printf("frame_period = %.1f ns, effective_channel_rate = %.1f MHz "
           "(= f_clk / n_channels = %.1f MHz / %d)\n",
           frame_period_rtl * 1e9, eff_rate_rtl / 1e6, f_clk_rtl / 1e6, n_ch_rtl);
    pass &= check_close("RTL-config frame_period == 40 ns", frame_period_rtl, 40e-9, 1e-15);
    pass &= check_close("RTL-config effective_rate == 25 MHz", eff_rate_rtl, 25e6, 1e-6);

    /* ---- the cycles/sec analysis: a real optical time-sharing scenario ---- */
    printf("\n=== Timing analysis: 1 GHz clock, 4 channels, 250 cycles/slot ===\n");
    tdm_config_t cfg = { .f_clk_hz = 1.0e9, .n_channels = 4, .cycles_per_slot = 250 };

    double frame_period = frame_period_s(cfg);
    double slot = slot_duration_s(cfg);
    double eff_rate = effective_channel_rate_hz(cfg);

    printf("f_clk = %.3e Hz, n_channels = %d, cycles_per_slot = %d\n",
           cfg.f_clk_hz, cfg.n_channels, cfg.cycles_per_slot);
    printf("frame_period          = %.6e s\n", frame_period);
    printf("slot_duration         = %.6e s\n", slot);
    printf("effective_channel_rate= %.6e Hz\n", eff_rate);

    printf("channel at the START of each slot, 16 slots: ");
    for (int i = 0; i < 16; i++)
        printf("%d ", channel_at_cycle(cfg, (long)i * cfg.cycles_per_slot));
    printf("\n\n");

    /* identity: frame_period == n_channels * slot_duration */
    pass &= check_close("frame_period == n_channels*slot_duration",
                         frame_period, cfg.n_channels * slot, 1e-15);

    /* identity: effective_channel_rate == f_clk / (n_channels*cycles_per_slot) */
    pass &= check_close("effective_rate == f_clk/(n*cycles_per_slot)",
                         eff_rate, cfg.f_clk_hz / (cfg.n_channels * cfg.cycles_per_slot), 1e-9);

    /* channel_at_cycle with cycles_per_slot=1 reduces exactly to tdm_active_channel */
    tdm_config_t cfg_unit_slot = { .f_clk_hz = 1.0, .n_channels = 4, .cycles_per_slot = 1 };
    int reduction_ok = 1;
    for (long c = 0; c < 16; c++)
        if (channel_at_cycle(cfg_unit_slot, c) != tdm_active_channel(c, 4)) reduction_ok = 0;
    printf("%-46s %s\n", "channel_at_cycle(slot=1) == tdm_active_channel",
           reduction_ok ? "OK" : "FAIL");
    pass &= reduction_ok;

    /* mid-slot cycle still belongs to the same channel as the slot's start */
    pass &= check("mid-slot cycle still belongs to the same channel",
                   channel_at_cycle(cfg, cfg.cycles_per_slot + cfg.cycles_per_slot / 2), 1);

    /* fairness: every channel gets EXACTLY cycles_per_slot cycles per frame */
    long counts[4] = {0, 0, 0, 0};
    long total_cycles = (long)cfg.n_channels * cfg.cycles_per_slot;
    for (long c = 0; c < total_cycles; c++)
        counts[channel_at_cycle(cfg, c)]++;
    int fair = 1;
    for (int ch = 0; ch < cfg.n_channels; ch++)
        if (counts[ch] != cfg.cycles_per_slot) fair = 0;
    printf("%-46s %s\n", "round-robin fairness over one frame", fair ? "OK" : "FAIL");
    pass &= fair;

    printf("\n%s\n", pass ? "ALL TESTS PASS" : "TESTS FAILED");
    return pass ? 0 : 1;
}
