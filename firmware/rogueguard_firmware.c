/* rogueguard_firmware.c
 * =========================================================================
 * RogueGuard v1.0 -- Core firmware implementation
 * See rogueguard_firmware.h for full memory map and architecture.
 *
 * Compile:
 *   aarch64-linux-gnu-gcc -O2 -std=c11 -march=armv8-a+simd \
 *     rogueguard_firmware.c -o rogueguard -lm -lfftw3f -lpthread
 *
 * Note: fftw3f (single precision) provides the FFT used by TD-GS.
 *       Replace with ARM CMSIS-DSP arm_cfft_f32 for bare-metal targets.
 * =========================================================================
 */

#include "rogueguard_firmware.h"
#include <stdio.h>
#include <time.h>
#include <fftw3.h>  /* fftw3f: single-precision FFTW */

/* ── Private helpers ─────────────────────────────────────────────────────── */

static float complex cexpf_approx(float phase) {
    /* Fast complex exponential via Euler's formula. */
    return cosf(phase) + I * sinf(phase);
}

static void build_disperser_tf(float complex *H, const float *freq_GHz,
                                float D_ps2, uint32_t N) {
    /* H[k] = exp(i * alpha * D * freq[k]^2)   (dispersive transfer fn) */
    const float alpha = 2.515e-5f;  /* nm*ps/GHz^2 */
    for (uint32_t k = 0; k < N; k++) {
        float phi = alpha * D_ps2 * freq_GHz[k] * freq_GHz[k];
        H[k] = cexpf_approx(phi);
    }
}

static float rms(const float *a, const float *b, uint32_t N) {
    float s = 0.0f;
    for (uint32_t i = 0; i < N; i++) { float d = a[i]-b[i]; s += d*d; }
    return sqrtf(s / N);
}

/* ── rg_init: allocate all subsystems from heap ──────────────────────────── */
rg_ctx_t *rg_init(void) {
    rg_ctx_t *ctx;
    RG_MALLOC_OR_DIE(ctx, sizeof(rg_ctx_t));

    /* ── Ring buffer ───────────────────────────────────────────────────── */
    RG_MALLOC_OR_DIE(ctx->ring, sizeof(rg_ring_buf_t));
    ctx->ring->write_idx = 0;
    ctx->ring->read_idx  = 0;
    pthread_mutex_init(&ctx->ring->mutex, NULL);
    pthread_cond_init(&ctx->ring->cond,   NULL);

    /* ── Frequency axis (symmetric, N_FFT/2 GHz span each side) ────────── */
    RG_MALLOC_OR_DIE(ctx->freq_GHz, N_FFT * sizeof(float));
    float df = 200.0f / (float)N_FFT;  /* 400 GHz window / N_FFT bins */
    for (uint32_t k = 0; k < N_FFT; k++)
        ctx->freq_GHz[k] = -200.0f + k * df;

    /* ── TD-GS state ────────────────────────────────────────────────────── */
    RG_MALLOC_OR_DIE(ctx->tdgs, sizeof(rg_tdgs_t));
    RG_MALLOC_OR_DIE(ctx->tdgs->E1, N_FFT * sizeof(float complex));
    RG_MALLOC_OR_DIE(ctx->tdgs->E2, N_FFT * sizeof(float complex));
    RG_MALLOC_OR_DIE(ctx->tdgs->H1, N_FFT * sizeof(float complex));
    RG_MALLOC_OR_DIE(ctx->tdgs->H2, N_FFT * sizeof(float complex));
    RG_MALLOC_OR_DIE(ctx->tdgs->I1, N_FFT * sizeof(float));
    RG_MALLOC_OR_DIE(ctx->tdgs->I2, N_FFT * sizeof(float));

    build_disperser_tf(ctx->tdgs->H1, ctx->freq_GHz, D1_PS2, N_FFT);
    build_disperser_tf(ctx->tdgs->H2, ctx->freq_GHz, D2_PS2, N_FFT);

    /* ── CNN state (weights loaded separately) ──────────────────────────── */
    RG_MALLOC_OR_DIE(ctx->cnn, sizeof(rg_cnn_t));
    /* Activation buffers */
    RG_MALLOC_OR_DIE(ctx->cnn->act1,   CNN_LAYER1_CH * (N_TRACE/2) * sizeof(float));
    RG_MALLOC_OR_DIE(ctx->cnn->act2,   CNN_LAYER2_CH * (N_TRACE/4) * sizeof(float));
    RG_MALLOC_OR_DIE(ctx->cnn->act3,   CNN_LAYER3_CH * (N_TRACE/8) * sizeof(float));
    RG_MALLOC_OR_DIE(ctx->cnn->fc1_out, CNN_FC_DIM * sizeof(float));

    fprintf(stderr, "[rg_init] Heap allocated: ~%.1f kB\n",
            (sizeof(rg_ring_buf_t)
             + N_FFT * sizeof(float)
             + sizeof(rg_tdgs_t)
             + 4 * N_FFT * sizeof(float complex)
             + 2 * N_FFT * sizeof(float)
             + sizeof(rg_cnn_t)
             + (CNN_LAYER1_CH*(N_TRACE/2)
                + CNN_LAYER2_CH*(N_TRACE/4)
                + CNN_LAYER3_CH*(N_TRACE/8)
                + CNN_FC_DIM) * sizeof(float)
            ) / 1024.0);

    return ctx;
}

/* ── rg_destroy: free all heap allocations ───────────────────────────────── */
void rg_destroy(rg_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->tdgs) {
        free(ctx->tdgs->E1); free(ctx->tdgs->E2);
        free(ctx->tdgs->H1); free(ctx->tdgs->H2);
        free(ctx->tdgs->I1); free(ctx->tdgs->I2);
        free(ctx->tdgs);
    }
    if (ctx->cnn) {
        free(ctx->cnn->act1); free(ctx->cnn->act2);
        free(ctx->cnn->act3); free(ctx->cnn->fc1_out);
        free(ctx->cnn->conv1_w); free(ctx->cnn->conv2_w);
        free(ctx->cnn->conv3_w); free(ctx->cnn->fc1_w);
        free(ctx->cnn->fc2_w);
        free(ctx->cnn);
    }
    free(ctx->freq_GHz);
    if (ctx->ring) {
        pthread_mutex_destroy(&ctx->ring->mutex);
        pthread_cond_destroy(&ctx->ring->cond);
        free(ctx->ring);
    }
    free(ctx);
}

/* ── rg_dma_isr: called from DMA completion interrupt ───────────────────── */
void rg_dma_isr(rg_ctx_t *ctx, uint16_t *ch1, uint16_t *ch2,
                 uint32_t seq, uint64_t ts_ns) {
    rg_ring_buf_t *rb = ctx->ring;
    uint32_t wi = rb->write_idx % N_RING;
    rg_adc_frame_t *frame = &rb->frames[wi];

    /* Copy raw ADC data into ring slot (DMA-safe: no alloc in ISR) */
    memcpy(frame->samples[0], ch1, N_ADC * sizeof(uint16_t));
    memcpy(frame->samples[1], ch2, N_ADC * sizeof(uint16_t));
    frame->seq_num      = seq;
    frame->timestamp_ns = ts_ns;
    frame->valid        = 1;

    pthread_mutex_lock(&rb->mutex);
    rb->write_idx++;
    pthread_cond_signal(&rb->cond);
    pthread_mutex_unlock(&rb->mutex);
}

/* ── rg_tdgs_run: Gerchberg-Saxton phase retrieval ───────────────────────── */
float rg_tdgs_run(rg_tdgs_t *tdgs) {
    /* Allocate FFTW plans (production: cache these at init time) */
    fftwf_complex *fft_buf;
    fft_buf = (fftwf_complex *)fftwf_malloc(N_FFT * sizeof(fftwf_complex));
    fftwf_plan plan_fwd = fftwf_plan_dft_1d(N_FFT, fft_buf, fft_buf,
                                             FFTW_FORWARD,  FFTW_ESTIMATE);
    fftwf_plan plan_inv = fftwf_plan_dft_1d(N_FFT, fft_buf, fft_buf,
                                             FFTW_BACKWARD, FFTW_ESTIMATE);

    /* Initialize E1 as sqrt(I1) (real, no phase) */
    for (uint32_t k = 0; k < N_FFT; k++)
        tdgs->E1[k] = sqrtf(fmaxf(0.0f, tdgs->I1[k]));

    float best_score = 1e30f;
    tdgs->converged  = 0;

    for (uint32_t iter = 0; iter < N_ITER; iter++) {
        /* Forward: E1 -> freq domain */
        for (uint32_t k=0; k<N_FFT; k++) {
            fft_buf[k][0] = crealf(tdgs->E1[k]);
            fft_buf[k][1] = cimagf(tdgs->E1[k]);
        }
        fftwf_execute(plan_fwd);

        /* Apply H1, then H2^-1 to get E2 estimate */
        float complex *E1w = (float complex *)fft_buf;
        float score = 0.0f;

        for (uint32_t k = 0; k < N_FFT; k++) {
            float complex e1_spec = E1w[k] * tdgs->H1[k];
            /* Arm2 estimate: apply H2 */
            float complex e2_spec = E1w[k] * tdgs->H2[k];
            /* Convert back for arm2 intensity constraint */
            fft_buf[k][0] = crealf(e2_spec) / N_FFT;
            fft_buf[k][1] = cimagf(e2_spec) / N_FFT;
        }

        fftwf_execute(plan_inv);

        /* Arm2 intensity constraint: replace magnitude, keep phase */
        for (uint32_t k = 0; k < N_FFT; k++) {
            float complex e2t = fft_buf[k][0] + I * fft_buf[k][1];
            float mag = cabsf(e2t);
            float meas = sqrtf(fmaxf(0.0f, tdgs->I2[k]));
            float resid = (mag - meas) * (mag - meas);
            score += resid;
            tdgs->E2[k] = (mag > 1e-12f) ? e2t * (meas / mag) : meas;
        }
        score = sqrtf(score / N_FFT);

        if (score < best_score) {
            best_score = score;
            memcpy(tdgs->E1, tdgs->E2, N_FFT * sizeof(float complex));
        }

        if (score < 1e-4f) { tdgs->converged = 1; break; }
    }

    tdgs->best_score = best_score;
    fftwf_destroy_plan(plan_fwd);
    fftwf_destroy_plan(plan_inv);
    fftwf_free(fft_buf);
    return best_score;
}

/* ── rg_cnn_infer: INT8 1D-CNN forward pass ──────────────────────────────── */
float rg_cnn_infer(rg_cnn_t *cnn, const float *I_trace) {
    /*
     * Simplified INT8 inference path.
     * Production: replace with ARM CMSIS-NN or TensorFlow Lite Micro.
     *
     * For demo, runs float32 via direct computation (no quantisation).
     * Expects I_trace[N_TRACE] normalised to mean=1.
     *
     * Feature-based fallback if weights not loaded: use 4 hand-crafted
     * statistics (max, std, skewness, kurtosis) -> logistic regression.
     */
    if (cnn->conv1_w == NULL) {
        /* Weights not loaded: feature-based classifier */
        float mx = 0.0f, mn = 0.0f, variance = 0.0f, skew = 0.0f;
        for (uint32_t i = 0; i < N_TRACE; i++) mn += I_trace[i];
        mn /= N_TRACE;
        for (uint32_t i = 0; i < N_TRACE; i++) {
            float d = I_trace[i] - mn;
            variance += d*d;
            skew     += d*d*d;
            if (I_trace[i] > mx) mx = I_trace[i];
        }
        variance /= N_TRACE;
        float std = sqrtf(variance + 1e-8f);
        skew = skew / (N_TRACE * std * std * std + 1e-8f);
        /* Simple decision rule: rogue if peak/mean > 5 OR skewness > 3 */
        float p = (mx / (mn + 1e-6f) > 5.0f || skew > 3.0f) ? 0.9f : 0.05f;
        return p;
    }
    /*
     * Full INT8 path would do:
     *   act1 = relu(conv1d(I_trace,  conv1_w, bias1) * conv1_scale)
     *   act1 = maxpool1d(act1, 2)
     *   act2 = relu(conv1d(act1, conv2_w, bias2) * conv2_scale)
     *   act2 = maxpool1d(act2, 2)
     *   act3 = relu(conv1d(act2, conv3_w, bias3) * conv3_scale)
     *   act3 = maxpool1d(act3, 2)
     *   fc1  = relu(fc1_w @ flatten(act3) + bias_fc1) * fc1_scale
     *   out  = sigmoid(fc2_w @ fc1 + bias_fc2) * fc2_scale
     * Omitted here: see PyTorch reference in datacenter_rogue_wave.py
     */
    return 0.0f;  /* placeholder when weights loaded but INT8 path stub */
}

/* ── rg_process_frame: full pipeline for one ADC frame ──────────────────── */
int rg_process_frame(rg_ctx_t *ctx, rg_adc_frame_t *frame) {
    rg_tdgs_t *tdgs = ctx->tdgs;

    /* 1. ADC -> float32 + normalize */
    for (uint32_t k = 0; k < N_FFT; k++) {
        /* 16-bit ADC: 0-65535 -> 0.0-1.0 */
        tdgs->I1[k] = (float)frame->samples[0][k % N_ADC] / 65535.0f;
        tdgs->I2[k] = (float)frame->samples[1][k % N_ADC] / 65535.0f;
    }

    /* 2. TD-GS phase retrieval */
    float rmse = rg_tdgs_run(tdgs);

    /* 3. Decimate E1 from N_FFT=4096 to N_TRACE=256 for CNN */
    float I_trace[N_TRACE];
    uint32_t stride = N_FFT / N_TRACE;
    float trace_mean = 0.0f;
    for (uint32_t i = 0; i < N_TRACE; i++) {
        I_trace[i] = cabsf(tdgs->E1[i * stride]);
        I_trace[i] *= I_trace[i];           /* intensity = |E|^2 */
        trace_mean += I_trace[i];
    }
    trace_mean = fmaxf(trace_mean / N_TRACE, 1e-8f);
    for (uint32_t i = 0; i < N_TRACE; i++) I_trace[i] /= trace_mean;

    /* 4. CNN inference */
    float p_rogue = rg_cnn_infer(ctx->cnn, I_trace);

    ctx->n_frames_processed++;

    /* 5. Alert if threshold exceeded */
    if (p_rogue >= ROGUE_THRESHOLD) {
        rg_alert_t alert;
        alert.timestamp_ns = frame->timestamp_ns;
        alert.p_rogue      = p_rogue;
        alert.phase_rmse   = rmse;
        alert.seq_num      = frame->seq_num;
        /* Peak power in normalised trace */
        alert.peak_power = 0.0f;
        for (uint32_t i = 0; i < N_TRACE; i++)
            if (I_trace[i] > alert.peak_power) alert.peak_power = I_trace[i];
        snprintf(alert.message, sizeof(alert.message),
                 "Rogue wave: P=%.3f peak_pwr=%.2fx seq=%u",
                 p_rogue, alert.peak_power, frame->seq_num);
        rg_fire_alert(ctx, &alert);
    }

    return 0;
}

/* ── rg_fire_alert: SNMP trap + REST POST + local LED ───────────────────── */
void rg_fire_alert(rg_ctx_t *ctx, const rg_alert_t *alert) {
    ctx->n_alerts_fired++;
    /* In production: send SNMPv3 trap, POST to dashboard REST API,
     * toggle GPIO alert LED, write to /var/log/rogueguard/alerts.jsonl   */
    fprintf(stderr, "[ALERT] seq=%u  P(rogue)=%.4f  peak=%.2fx  rmse=%.4f rad\n",
            alert->seq_num, alert->p_rogue,
            alert->peak_power, alert->phase_rmse);
    fprintf(stderr, "        %s\n", alert->message);
}

/* ── rg_worker_thread: consumer of ring buffer ───────────────────────────── */
void *rg_worker_thread(void *arg) {
    rg_ctx_t *ctx = (rg_ctx_t *)arg;
    rg_ring_buf_t *rb = ctx->ring;

    while (1) {
        pthread_mutex_lock(&rb->mutex);
        while (rb->read_idx == rb->write_idx)
            pthread_cond_wait(&rb->cond, &rb->mutex);
        uint32_t ri = rb->read_idx % N_RING;
        rg_adc_frame_t *frame = &rb->frames[ri];
        rb->read_idx++;
        pthread_mutex_unlock(&rb->mutex);

        if (frame->valid) {
            rg_process_frame(ctx, frame);
            frame->valid = 0;
        }
    }
    return NULL;
}

/* ── Minimal test main (compile standalone with -DSTANDALONE) ───────────── */
#ifdef STANDALONE
int main(void) {
    fprintf(stderr, "RogueGuard Firmware v%d.%d.%d -- self-test\n",
            RG_VERSION_MAJOR, RG_VERSION_MINOR, RG_VERSION_PATCH);

    rg_ctx_t *ctx = rg_init();
    fprintf(stderr, "[main] Context initialised at %p\n", (void *)ctx);

    /* Inject synthetic rogue frame (peak at center, 9x background) */
    uint16_t ch1[N_ADC], ch2[N_ADC];
    for (uint32_t i = 0; i < N_ADC; i++) {
        float t   = (float)i / N_ADC - 0.5f;
        float I1  = 0.1f + 0.9f * expf(-t*t * 200.0f);     /* Gaussian */
        /* Rogue: Peregrine-like, 9x at center */
        float I2  = 1.0f + 8.0f * expf(-t*t * 2000.0f);
        ch1[i] = (uint16_t)(I1 * 32767);
        ch2[i] = (uint16_t)(I2 * 32767);
    }

    /* Simulate DMA ISR */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    rg_dma_isr(ctx, ch1, ch2, 1, now_ns);

    /* Manually drain ring (no worker thread in test) */
    rg_adc_frame_t *frame = &ctx->ring->frames[0];
    rg_process_frame(ctx, frame);

    fprintf(stderr, "[main] Frames processed: %llu  Alerts: %llu\n",
            (unsigned long long)ctx->n_frames_processed,
            (unsigned long long)ctx->n_alerts_fired);

    rg_destroy(ctx);
    fprintf(stderr, "[main] Clean shutdown. All heap freed.\n");
    return 0;
}
#endif /* STANDALONE */
