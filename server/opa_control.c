/*
 * opa_control.c  —  Phase → OPA element voltages (DAC SPI)
 * Target: AWS Graviton 3 / RPi CM4, aarch64-linux
 *
 * Optical Phased Array (OPA) beamforming:
 *   - N_ELEMENTS phase actuators driven by SPI DAC (AD5628, 12-bit, 8-ch)
 *   - Phase commands φ[k] from GS solver → voltage Vk = Vπ · (φ[k] / π)
 *   - SPI device: /dev/spidev0.0, mode 0, 20 MHz
 *   - Latency target: < 0.5 ms for full 64-element array update
 *
 * Beamsteering:
 *   φ[k] = (2π/λ) · d · k · sin(θ)     d = 10 µm element pitch
 *   scan: θ ∈ [-30°, +30°] → φ ∈ [-π, π] per element
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

/* ── OPA parameters ───────────────────────────────────────────────── */
#define N_ELEMENTS        64          /* OPA element count                */
#define DAC_BITS          12          /* AD5628 resolution                */
#define DAC_FULL_SCALE    4095        /* 2^12 - 1                         */
#define V_PI_VOLTS        3.3f        /* half-wave voltage of phase shifter */
#define ELEMENT_PITCH_UM  10.0        /* element pitch [µm]               */
#define LAMBDA_NM         1550.0      /* operating wavelength [nm]        */

/* ── SPI configuration ────────────────────────────────────────────── */
#define SPI_DEVICE        "/dev/spidev0.0"
#define SPI_MODE          0
#define SPI_BITS          8
#define SPI_SPEED_HZ      20000000    /* 20 MHz                           */

/* Emulation flag: set 1 when no SPI hardware available */
#define OPA_EMULATE       1

typedef struct {
    int   spi_fd;
    int   emulate;
    float phase_cmd[N_ELEMENTS];   /* last commanded phase [rad] */
    float voltage[N_ELEMENTS];     /* last DAC voltage [V]       */
} OpaHandle;

/* ── Initialise OPA handle ────────────────────────────────────────── */
OpaHandle *opa_open(void)
{
    OpaHandle *h = calloc(1, sizeof(OpaHandle));
    if (!h) return NULL;

#if OPA_EMULATE
    h->emulate = 1;
    h->spi_fd  = -1;
    fprintf(stderr, "[opa_control] emulation mode (no SPI hardware)\n");
#else
    h->spi_fd = open(SPI_DEVICE, O_RDWR);
    if (h->spi_fd < 0) {
        perror("[opa_control] open SPI");
        free(h); return NULL;
    }
    /* Configure SPI mode, bits, speed via ioctl */
    uint8_t  mode  = SPI_MODE;
    uint8_t  bits  = SPI_BITS;
    uint32_t speed = SPI_SPEED_HZ;
    ioctl(h->spi_fd, SPI_IOC_WR_MODE,          &mode);
    ioctl(h->spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
    ioctl(h->spi_fd, SPI_IOC_WR_MAX_SPEED_HZ,  &speed);
    fprintf(stderr, "[opa_control] SPI %s opened @ %u Hz\n", SPI_DEVICE, speed);
#endif
    return h;
}

/* ── Phase → DAC code conversion ─────────────────────────────────── */
static uint16_t phase_to_dac(float phi)
{
    /* Wrap φ to [0, 2π) then map to [0, V_pi] voltage range.
     * Modulator: V → phase shift = π·V/Vπ */
    while (phi < 0.0f)    phi += 2.0f * (float)M_PI;
    while (phi >= 2.0f * (float)M_PI) phi -= 2.0f * (float)M_PI;

    /* Map [0, 2π) → [0, V_pi] */
    float v = V_PI_VOLTS * (phi / (float)M_PI);
    if (v > V_PI_VOLTS) v = V_PI_VOLTS;

    /* Quantise to DAC_BITS */
    int code = (int)(v / V_PI_VOLTS * DAC_FULL_SCALE + 0.5f);
    if (code < 0)             code = 0;
    if (code > DAC_FULL_SCALE) code = DAC_FULL_SCALE;
    return (uint16_t)code;
}

/* ── Write phase commands to all OPA elements ─────────────────────── */
int opa_write_phase(OpaHandle *h, const float *phi, int n)
{
    if (!h) return -1;
    int count = n < N_ELEMENTS ? n : N_ELEMENTS;

    /* Build SPI transfer buffer: 3 bytes per AD5628 write command
     *   Byte 0: [CMD(4) | ADDR(4)]   CMD=0x3 (write+update), ADDR = element index % 8
     *   Byte 1: MSB 8 bits of 12-bit code
     *   Byte 2: LSB 4 bits << 4
     */
    uint8_t txbuf[N_ELEMENTS * 3];
    for (int k = 0; k < count; k++) {
        h->phase_cmd[k] = phi[k];
        uint16_t code    = phase_to_dac(phi[k]);
        h->voltage[k]    = (float)code / DAC_FULL_SCALE * V_PI_VOLTS;

        txbuf[k*3 + 0] = 0x30 | (k & 0x07);     /* CMD=write+update, ADDR */
        txbuf[k*3 + 1] = (code >> 4) & 0xFF;
        txbuf[k*3 + 2] = (code & 0x0F) << 4;
    }

    if (h->emulate) {
        /* Print first 4 elements for debug */
        fprintf(stderr, "[opa_control] emulated write: φ[0..3] = "
                "%.3f %.3f %.3f %.3f rad → DAC %u %u %u %u\n",
                (double)phi[0], (double)phi[1],
                (double)phi[2], (double)phi[3],
                phase_to_dac(phi[0]), phase_to_dac(phi[1]),
                phase_to_dac(phi[2]), phase_to_dac(phi[3]));
        return 0;
    }

#ifndef OPA_EMULATE
    /* Full SPI burst write via ioctl SPI_IOC_MESSAGE */
    struct spi_ioc_transfer xfer = {
        .tx_buf        = (unsigned long)txbuf,
        .rx_buf        = 0,
        .len           = count * 3,
        .speed_hz      = SPI_SPEED_HZ,
        .bits_per_word = SPI_BITS,
        .cs_change     = 0,
    };
    if (ioctl(h->spi_fd, SPI_IOC_MESSAGE(1), &xfer) < 0) {
        perror("[opa_control] SPI write"); return -1;
    }
#endif
    return 0;
}

/* ── Beamsteering: steer beam to angle θ [degrees] ───────────────── */
int opa_steer(OpaHandle *h, double theta_deg)
{
    double theta = theta_deg * M_PI / 180.0;
    double lambda_m = LAMBDA_NM * 1e-9;
    double d_m      = ELEMENT_PITCH_UM * 1e-6;
    float  phi[N_ELEMENTS];

    for (int k = 0; k < N_ELEMENTS; k++) {
        double phase = (2.0 * M_PI / lambda_m) * d_m * k * sin(theta);
        /* Wrap to [-π, π] */
        phase = fmod(phase, 2.0 * M_PI);
        if (phase >  M_PI) phase -= 2.0 * M_PI;
        if (phase < -M_PI) phase += 2.0 * M_PI;
        phi[k] = (float)phase;
    }

    fprintf(stderr, "[opa_control] steer θ=%.1f° → φ[0]=%.3f φ[N-1]=%.3f rad\n",
            theta_deg, (double)phi[0], (double)phi[N_ELEMENTS-1]);
    return opa_write_phase(h, phi, N_ELEMENTS);
}

/* ── Close handle ─────────────────────────────────────────────────── */
void opa_close(OpaHandle *h)
{
    if (!h) return;
    if (h->spi_fd >= 0) close(h->spi_fd);
    free(h);
}
