/* rogueguard_firmware.h
 * =========================================================================
 * RogueGuard v1.0 -- Embedded Firmware for 1U Optical Rogue Wave Monitor
 * Target:  Raspberry Pi CM4 (Cortex-A72, 4 GB LPDDR4, 32 GB eMMC)
 *          Custom ADC HAT: dual-channel 14-bit, 56 GSa/s (clocked)
 * Toolchain: aarch64-linux-gnu-gcc -O2 -std=c11 -lm -lfftw3f -lpthread
 *
 * MEMORY MAP (16 MB firmware region in DRAM):
 *
 *   Base address    Size    Region
 *   0x10000000     32 kB   .text + .rodata (code, constants)
 *   0x10008000     16 kB   .data + .bss (globals)
 *   0x1000C000     32 kB   Stack (8 kB x 4 Cortex-A72 cores)
 *   0x10014000    448 kB   Heap (malloc arena, dlmalloc)
 *     0x10014000   16 kB     ADC ring buffer  (2 ch x N_ADC x uint16_t x N_RING)
 *     0x10018000   64 kB     FFT workspace    (2 x N_FFT x complex float)
 *     0x10028000  128 kB     TD-GS state      (E1/E2/H1/H2 arrays x N_FFT)
 *     0x10048000   32 kB     CNN activations  (per-layer activation buffers)
 *     0x10050000  200 kB     CNN weights      (INT8-quantized, loaded from eMMC)
 *     0x10082000    8 kB     I/O message queue (alert events, dashboard frames)
 *   0x10090000   ...        Available for OS + userspace
 *
 * Data flow (real-time pipeline, locked to ADC interrupt):
 *   ADC ISR -> DMA -> ring_buf -> worker thread:
 *     [decimate 56G->56M] -> [rfftw3] -> [TD-GS 200 iter] -> [CNN] -> [alert]
 *
 * Latency budget:
 *   ADC window N_ADC=4096 at 56 GSa/s: 73 ns acquisition
 *   Decimation 56G->56M (factor 1000): software, 0.1 ms
 *   FFT (N_FFT=4096, FFTW3 single):   0.08 ms
 *   TD-GS (200 iter, vectorised):      1.2 ms
 *   CNN inference (INT8, N_TRACE=256): 0.05 ms
 *   Total pipeline:                    ~1.5 ms  (<< 10 ms alert SLA)
 * =========================================================================
 */

#ifndef ROGUEGUARD_FIRMWARE_H
#define ROGUEGUARD_FIRMWARE_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>

/* ── Constants ────────────────────────────────────────────────────────────── */
#define RG_VERSION_MAJOR  1
#define RG_VERSION_MINOR  0
#define RG_VERSION_PATCH  0

#define N_ADC       4096U   /* ADC samples per acquisition window           */
#define N_FFT       4096U   /* FFT points (must be power of 2)              */
#define N_TRACE     256U    /* CNN input length (decimated)                 */
#define N_ITER      200U    /* TD-GS max iterations                         */
#define N_RING      8U      /* ring buffer depth (frames)                   */
#define N_CH        2U      /* number of ADC channels (D1, D2)              */

#define ADC_RATE_GHZ   56.0f       /* ADC sample rate (GHz)                */
#define DECIM_FACTOR   1000U       /* decimation: 56 GHz -> 56 MHz          */
#define ALPHA_CO       2.515e-5f   /* dispersion coefficient nm*ps/GHz^2   */
#define D1_PS2         -600.0f     /* disperser 1 GDD (ps^2)               */
#define D2_PS2         -900.0f     /* disperser 2 GDD (ps^2)               */

#define ROGUE_THRESHOLD  0.5f      /* P(rogue) threshold for SNMP alert     */
#define ALERT_COOLDOWN_MS 100      /* min ms between successive alerts      */

/* ── ADC ring buffer ─────────────────────────────────────────────────────── */
typedef struct {
    uint16_t  samples[N_CH][N_ADC]; /* raw ADC samples, interleaved ch     */
    uint32_t  seq_num;              /* acquisition sequence number          */
    uint64_t  timestamp_ns;         /* system timestamp (CLOCK_REALTIME)    */
    uint8_t   valid;                /* 1 = data ready for processing        */
} rg_adc_frame_t;

typedef struct {
    rg_adc_frame_t  frames[N_RING]; /* circular buffer of ADC frames       */
    volatile uint32_t write_idx;    /* written by DMA ISR                  */
    volatile uint32_t read_idx;     /* consumed by worker thread           */
    pthread_mutex_t   mutex;
    pthread_cond_t    cond;
} rg_ring_buf_t;

/* ── TD-GS phase retrieval state ─────────────────────────────────────────── */
typedef struct {
    float complex  *E1;            /* field in arm 1 (N_FFT complex floats) */
    float complex  *E2;            /* field in arm 2                        */
    float complex  *H1;            /* disperser transfer fn arm 1           */
    float complex  *H2;            /* disperser transfer fn arm 2           */
    float          *I1;            /* measured intensity arm 1 (N_FFT)      */
    float          *I2;            /* measured intensity arm 2              */
    float           best_score;    /* best residual achieved                */
    uint32_t        iters_run;     /* iterations actually executed          */
    uint8_t         converged;     /* 1 if tol reached                      */
} rg_tdgs_t;

/* ── CNN inference state (INT8 quantized) ────────────────────────────────── */
#define CNN_LAYER1_CH  16
#define CNN_LAYER2_CH  32
#define CNN_LAYER3_CH  64
#define CNN_FC_DIM    128

typedef struct {
    /* Quantized weights (loaded from eMMC /data/cnn_weights.bin)          */
    int8_t  *conv1_w;   /* (16, 1,  7) */
    int8_t  *conv2_w;   /* (32, 16, 5) */
    int8_t  *conv3_w;   /* (64, 32, 3) */
    int8_t  *fc1_w;     /* (128, 64*32) */
    int8_t  *fc2_w;     /* (1,  128)   */
    float    conv1_scale, conv2_scale, conv3_scale, fc1_scale, fc2_scale;
    /* Activation buffers (reused across inferences) */
    float   *act1;      /* (16, N_TRACE/2)  */
    float   *act2;      /* (32, N_TRACE/4)  */
    float   *act3;      /* (64, N_TRACE/8)  */
    float   *fc1_out;   /* (128,)           */
} rg_cnn_t;

/* ── Alert event ─────────────────────────────────────────────────────────── */
typedef struct {
    uint64_t  timestamp_ns;
    float     p_rogue;        /* CNN output probability                     */
    float     phase_rmse;     /* TD-GS phase RMSE (rad)                     */
    float     peak_power;     /* normalised peak power in I(t)              */
    uint32_t  seq_num;        /* ADC frame that triggered alert             */
    char      message[128];   /* human-readable description                 */
} rg_alert_t;

/* ── System context (single global instance, malloc'd at boot) ───────────── */
typedef struct {
    rg_ring_buf_t  *ring;     /* ADC ring buffer                            */
    rg_tdgs_t      *tdgs;     /* TD-GS state                                */
    rg_cnn_t       *cnn;      /* CNN inference state                        */
    float          *freq_GHz; /* frequency axis (N_FFT points)              */
    uint64_t        n_frames_processed;
    uint64_t        n_alerts_fired;
    uint64_t        uptime_ns;
    pthread_t       worker_tid;
} rg_ctx_t;

/* ── Function prototypes ─────────────────────────────────────────────────── */
rg_ctx_t      *rg_init(void);
void           rg_destroy(rg_ctx_t *ctx);
void           rg_dma_isr(rg_ctx_t *ctx, uint16_t *ch1, uint16_t *ch2,
                           uint32_t seq, uint64_t ts_ns);
int            rg_process_frame(rg_ctx_t *ctx, rg_adc_frame_t *frame);
float          rg_tdgs_run(rg_tdgs_t *tdgs);
float          rg_cnn_infer(rg_cnn_t *cnn, const float *I_trace);
void           rg_fire_alert(rg_ctx_t *ctx, const rg_alert_t *alert);
void          *rg_worker_thread(void *arg);
int            rg_load_cnn_weights(rg_cnn_t *cnn, const char *path);

/* ── Utility macros ──────────────────────────────────────────────────────── */
#define RG_ASSERT(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "ASSERT FAIL: %s\n", msg); abort(); } } while(0)

#define RG_MALLOC_OR_DIE(ptr, n_bytes) \
    do { (ptr) = malloc(n_bytes); \
         RG_ASSERT((ptr) != NULL, "malloc failed: " #ptr); \
         memset((ptr), 0, n_bytes); } while(0)

#endif /* ROGUEGUARD_FIRMWARE_H */
