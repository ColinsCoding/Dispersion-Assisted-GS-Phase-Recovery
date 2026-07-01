"""Edge AI, wireless, nuclear detection, and communication protocols.

WHAT WAS MISSING FROM dgs/circuits.py:
  1. Edge AI inference circuits  -- TinyML, quantization, INT8 MAC units
  2. Wireless front-end          -- RF mixer, LNA, ADC chain, link budget
  3. Nuclear/radiation detection -- Geiger-Muller tube, scintillator + SiPM
  4. Communication protocols     -- UART, SPI, I2C, CAN (the real embedded stack)

JALALI CONNECTION:
  STEAM ADC board = RF front-end (LNA + mixer) + photonic stretch + ADC.
  Edge AI = running the GS phase retrieval on-chip (FPGA or ARM Cortex-M).
  Nuclear detection = SiPM (Silicon PhotoMultiplier) has same circuit as
    STEAM photodetector: transimpedance amplifier (TIA) + ADC.
  Protocols = the STEAM board communicates with host PC via SPI or PCIe.

C FILE I/O (open a file in C):
  Every protocol implementation starts with a file descriptor.
  In POSIX C (Linux/RPi): fd = open("data.bin", O_RDWR | O_CREAT, 0644)
  This is the same syscall used to open SPI device: open("/dev/spidev0.0", ...)

Run: py -3.13 -c "from dgs.edge_ai_protocols import demo; demo()"
"""
import numpy as np


# ── C FILE I/O (the missing piece) ───────────────────────────────────────────

C_FILE_IO = """
/* ── C file I/O: open, read, write, close ── */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>      /* open() flags: O_RDONLY, O_WRONLY, O_RDWR, O_CREAT */
#include <unistd.h>     /* read(), write(), close() */
#include <sys/stat.h>   /* mode bits: S_IRUSR, S_IWUSR */

/* ── POSIX low-level I/O (used for devices too) ── */
int posix_example(void) {
    int fd;
    ssize_t n;
    char buf[256];

    /* open(): returns file descriptor (int >= 0) or -1 on error */
    fd = open("steam_data.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return -1; }

    /* write(): returns bytes written or -1 */
    float sample = 3.14159f;
    n = write(fd, &sample, sizeof(float));   /* raw binary write */

    /* lseek() to rewind */
    lseek(fd, 0, SEEK_SET);

    /* read(): returns bytes read, 0 = EOF, -1 = error */
    float readback;
    n = read(fd, &readback, sizeof(float));

    close(fd);
    return 0;
}

/* ── stdio high-level (buffered, portable) ── */
int stdio_example(void) {
    FILE *fp = fopen("output.csv", "w");
    if (!fp) { perror("fopen"); return -1; }

    fprintf(fp, "time_ps,intensity\\n");
    for (int i = 0; i < 1024; i++) {
        fprintf(fp, "%d,%.6f\\n", i, (float)i / 1024.0f);
    }
    fclose(fp);

    /* read back */
    fp = fopen("output.csv", "r");
    char line[128];
    while (fgets(line, sizeof(line), fp) != NULL) {
        /* process line */
    }
    fclose(fp);
    return 0;
}

/* ── SPI device file (same open() syscall as regular file) ── */
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>

int spi_open_example(void) {
    /* SPI device is a FILE -- same fd = open() interface */
    int fd = open("/dev/spidev0.0", O_RDWR);
    if (fd < 0) { perror("SPI open"); return -1; }

    uint32_t mode = SPI_MODE_0;
    uint32_t speed = 10000000;   /* 10 MHz */
    ioctl(fd, SPI_IOC_WR_MODE32, &mode);
    ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);

    /* Now read/write to SPI just like a regular file */
    uint8_t tx[4] = {0x01, 0x02, 0x03, 0x04};
    uint8_t rx[4] = {0};
    struct spi_ioc_transfer tr = {
        .tx_buf = (uint64_t)tx,
        .rx_buf = (uint64_t)rx,
        .len = 4,
        .speed_hz = speed,
        .bits_per_word = 8,
    };
    ioctl(fd, SPI_IOC_MESSAGE(1), &tr);
    close(fd);
    return 0;
}

/* KEY INSIGHT: in POSIX, EVERYTHING is a file descriptor.
   Regular file, SPI device, I2C bus, GPIO, network socket --
   all opened with open(), read/written with read()/write(), closed with close().
   This is why C is the language of embedded systems. */
"""

C_ADR_EXPLANATION = """
ADR = Address Register (or Address in memory/protocol context)

In C memory:
  int arr[10];
  int *ptr = arr;        /* ptr holds the ADDRESS of arr[0] */
  int *adr = &arr[5];   /* adr = address of arr[5] */
  *adr = 42;            /* dereference: write 42 to the address */

In SPI protocol:
  ADR = the register address you send first before data.
  Frame: [ADR | R/W bit][DATA byte 0][DATA byte 1]...
  Example: write 0x1A to register 0x42 on an ADC:
    tx[0] = 0x42;   /* ADR = register 0x42 */
    tx[1] = 0x1A;   /* DATA */
    ioctl(fd, SPI_IOC_MESSAGE(1), &tr);

In I2C protocol:
  Device address (7-bit) + register address (8-bit) + data.
  i2c_smbus_write_byte_data(fd, reg_addr, value);
"""


# ── 1. Edge AI inference circuit ─────────────────────────────────────────────

def int8_mac_energy(n_macs, freq_GHz=1.0, E_mac_pJ=0.038):
    """Energy per inference for INT8 MAC array (TinyML / Edge AI).

    Edge AI = running neural network AT the sensor, NOT in the cloud.
    Motivation: STEAM generates 10 GB/s -- too much to send to cloud.
    Solution: on-chip INT8 neural net classifies each cell in real time.

    INT8 quantization: weights stored as 8-bit integers (not 32-bit float).
    Energy per MAC (multiply-accumulate): ~0.038 pJ at 28nm CMOS (Google TPU).
    Compare: FP32 MAC = 3.7 pJ (100x more expensive).

    TinyML: ARM Cortex-M55 + Ethos-U55 NPU -> 256 TOPS/W.
    """
    E_inference_pJ = n_macs * E_mac_pJ
    E_inference_uJ = E_inference_pJ * 1e-6
    power_mW = E_inference_uJ * 1e-6 * freq_GHz * 1e9 * 1e3
    throughput_inf_per_sec = freq_GHz * 1e9 / n_macs
    return {
        "n_macs": n_macs,
        "E_per_mac_pJ": E_mac_pJ,
        "E_inference_pJ": round(E_inference_pJ, 2),
        "power_mW": round(power_mW, 3),
        "throughput_fps": round(throughput_inf_per_sec),
        "vs_fp32": f"INT8 = {3.7/E_mac_pJ:.0f}x less energy than FP32",
        "jalali_use": "Classify STEAM cell images at 36 Mfps on Cortex-M NPU",
    }


def quantize_weights(weights_fp32, bits=8):
    """Quantize FP32 weights to INT8 for edge inference.

    scale = (max - min) / (2^bits - 1)
    zero_point = round(-min / scale)
    w_int = clip(round(w / scale) + zero_point, -128, 127)

    Dequantize: w_fp32 = (w_int - zero_point) * scale
    """
    w = np.asarray(weights_fp32, float)
    w_min, w_max = w.min(), w.max()
    n_levels = 2**bits - 1
    scale = (w_max - w_min) / n_levels if w_max != w_min else 1.0
    zero_point = int(np.round(-w_min / scale))
    w_int = np.clip(np.round(w / scale) + zero_point, -(2**(bits-1)), 2**(bits-1)-1).astype(int)
    w_dequant = (w_int - zero_point) * scale
    quant_error = np.max(np.abs(w - w_dequant))
    return {"w_int8": w_int, "scale": round(scale, 6),
            "zero_point": zero_point,
            "max_quant_error": round(quant_error, 6),
            "compression": f"FP32 -> INT8: {32//bits}x smaller model"}


# ── 2. Wireless RF front-end ─────────────────────────────────────────────────

def rf_link_budget(P_tx_dBm, G_tx_dBi, G_rx_dBi, freq_GHz,
                   distance_m, NF_dB=5.0, BW_MHz=20.0, SNR_req_dB=10.0):
    """Wireless link budget: can the signal be received?

    Friis transmission equation:
      P_rx = P_tx + G_tx + G_rx - FSPL
      FSPL = 20*log10(4*pi*d*f/c)  [free-space path loss, dB]

    Noise floor: N = kT*B*NF = -174 + 10*log10(BW_Hz) + NF  [dBm]
    Required P_rx >= N + SNR_required

    STEAM wireless: transmit processed phase data over 5G/WiFi to host.
    """
    FSPL_dB = 20*np.log10(4*np.pi*distance_m*freq_GHz*1e9/2.998e8)
    P_rx_dBm = P_tx_dBm + G_tx_dBi + G_rx_dBi - FSPL_dB
    noise_floor_dBm = -174.0 + 10*np.log10(BW_MHz*1e6) + NF_dB
    SNR_rx_dB = P_rx_dBm - noise_floor_dBm
    margin_dB = SNR_rx_dB - SNR_req_dB
    return {
        "P_rx_dBm": round(P_rx_dBm, 2),
        "FSPL_dB": round(FSPL_dB, 2),
        "noise_floor_dBm": round(noise_floor_dBm, 2),
        "SNR_rx_dB": round(SNR_rx_dB, 2),
        "link_margin_dB": round(margin_dB, 2),
        "link_ok": margin_dB > 0,
        "freq_GHz": freq_GHz,
        "distance_m": distance_m,
    }


def lna_noise_figure(G_dB, NF_dB, T_K=290):
    """Low Noise Amplifier: noise figure and effective noise temperature.

    Friis noise formula for cascade:
      F_total = F1 + (F2-1)/G1 + (F3-1)/(G1*G2) + ...

    F = noise factor (linear), NF = 10*log10(F) [dB]
    T_eff = (F-1) * T_0  [noise temperature, T_0=290K standard]

    LNA MUST be first in the chain: highest NF stage after G degrades SNR most.
    """
    F = 10**(NF_dB/10)
    G = 10**(G_dB/10)
    T_eff = (F - 1) * T_K
    return {"G_dB": G_dB, "NF_dB": NF_dB, "G_linear": round(G, 2),
            "F_linear": round(F, 4), "T_eff_K": round(T_eff, 2),
            "rule": "LNA first: low NF stage before high-G stages minimizes total noise"}


# ── 3. Nuclear / radiation detection ─────────────────────────────────────────

def geiger_muller_circuit(R_anode_ohm=1e6, C_coupling_pF=100e-12,
                          V_bias=400.0, t_dead_us=100.0):
    """Geiger-Muller tube + quench circuit model.

    GM tube: gas-filled diode, biased at 400-900V.
    When ionizing radiation (alpha/beta/gamma) enters:
      1. Ion pair created in gas
      2. Avalanche multiplication -> current pulse
      3. Anode voltage drops -> quenches discharge
      4. Dead time t_dead ~ 100 us (can't detect next event until recharged)

    Circuit: High voltage source -> R_anode -> GM tube -> C_coupling -> amp

    CORRECTION FOR DEAD TIME (Geiger statistics):
      True rate N_true = N_measured / (1 - N_measured * t_dead)

    PHOTONIC CONNECTION:
      SiPM (Silicon PhotoMultiplier) = CMOS Geiger mode APD array.
      Same avalanche physics, but solid-state, 1/10 the voltage.
      STEAM photodetector uses InGaAs APD (same avalanche principle).
    """
    tau_RC = R_anode_ohm * C_coupling_pF
    f_3dB = 1.0 / (2 * np.pi * tau_RC)
    max_rate_Hz = 1.0 / (t_dead_us * 1e-6)
    return {
        "V_bias_V": V_bias,
        "tau_RC_ns": round(tau_RC * 1e9, 2),
        "f_3dB_kHz": round(f_3dB / 1e3, 2),
        "max_count_rate_Hz": round(max_rate_Hz),
        "dead_time_us": t_dead_us,
        "dead_time_correction": "N_true = N_obs / (1 - N_obs * t_dead)",
        "sipm_connection": "SiPM = CMOS GM-mode APD; same physics as STEAM InGaAs APD",
    }


def radiation_dose(activity_Bq, distance_m, gamma_energy_MeV=1.17,
                   exposure_hours=1.0):
    """Rough gamma dose rate (point source, air, no shielding).

    Dose rate [mSv/h] ~ 0.5 * A [MBq] * E [MeV] / d^2 [m^2]
    (simplified; ignores buildup, scatter, attenuation)

    For Co-60 (E=1.17 + 1.33 MeV, common calibration source):
    At 1m from 1 MBq source: ~0.5 mSv/h  (NRC limit: 5 mSv/year occupational)
    """
    A_MBq = activity_Bq / 1e6
    dose_rate_mSv_h = 0.5 * A_MBq * gamma_energy_MeV / (distance_m**2)
    dose_mSv = dose_rate_mSv_h * exposure_hours
    nrc_annual_limit_mSv = 50.0
    return {
        "dose_rate_mSv_h": round(dose_rate_mSv_h, 4),
        "dose_mSv": round(dose_mSv, 4),
        "NRC_annual_limit_mSv": nrc_annual_limit_mSv,
        "fraction_annual_limit": round(dose_mSv / nrc_annual_limit_mSv, 6),
        "safe": dose_mSv < 1.0,
    }


# ── 4. Communication protocols ────────────────────────────────────────────────

def uart_frame(data_byte, baud=115200, parity="none", stop_bits=1):
    """UART frame timing and bit structure.

    UART frame: [START][D0 D1 D2 D3 D4 D5 D6 D7][PARITY?][STOP]
    START bit: logic LOW (break from idle HIGH)
    STOP bit:  logic HIGH (return to idle)
    Baud rate: bits per second

    Used for: debug console on RPi CM4 (RogueGuard), GPS modules, sensors.
    In C: open("/dev/ttyS0", O_RDWR); tcsetattr(fd, TCSANOW, &tty);
    """
    t_bit_us = 1e6 / baud
    n_bits = 1 + 8 + (1 if parity != "none" else 0) + stop_bits
    frame_us = n_bits * t_bit_us
    bits = []
    bits.append(0)   # start
    for i in range(8):
        bits.append((data_byte >> i) & 1)
    if parity == "even":
        bits.append(bin(data_byte).count('1') % 2)
    elif parity == "odd":
        bits.append((bin(data_byte).count('1') + 1) % 2)
    bits.append(1)   # stop
    return {"baud": baud, "t_bit_us": round(t_bit_us, 4),
            "frame_us": round(frame_us, 4), "n_bits": n_bits,
            "bits": bits, "hex": hex(data_byte),
            "c_device": "/dev/ttyS0 (Linux) or COM3 (Windows)"}


def spi_frame(address, data_bytes, cpol=0, cpha=0, cs_active=0):
    """SPI frame: address + data, full-duplex.

    SPI wires: MOSI (master out), MISO (master in), SCK (clock), CS (chip select)
    CPOL: clock polarity (0=idle low, 1=idle high)
    CPHA: clock phase (0=sample on leading edge, 1=sample on trailing edge)

    ADC SPI typical: CPOL=0, CPHA=0 (Mode 0), 10 MHz max.
    STEAM ADC (LTC2387): SPI + LVDS, 18-bit, 15 MSPS.

    C device: fd = open("/dev/spidev0.0", O_RDWR)
              ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed)
    """
    frame = [address] + list(data_bytes)
    mode = (cpol << 1) | cpha
    return {"frame_bytes": [hex(b) for b in frame],
            "n_bytes": len(frame),
            "mode": f"SPI Mode {mode} (CPOL={cpol}, CPHA={cpha})",
            "cs_active_level": cs_active,
            "c_device": "/dev/spidev0.0",
            "ltc2387_note": "18-bit STEAM ADC: SPI config + LVDS data output"}


def i2c_transaction(device_addr_7bit, reg_addr, data=None):
    """I2C write (or read if data=None) transaction.

    I2C frame (write): [START][ADDR<<1|0][ACK][REG][ACK][DATA][ACK][STOP]
    I2C frame (read):  [START][ADDR<<1|0][ACK][REG][ACK]
                       [START][ADDR<<1|1][ACK][DATA][NACK][STOP]

    Speed: Standard 100 kHz, Fast 400 kHz, Fast+ 1 MHz, HS 3.4 MHz.
    Used for: temperature sensors, IMU (accelerometer), DAC, EEPROM on RPi CM4.

    C: fd = open("/dev/i2c-1", O_RDWR)
       ioctl(fd, I2C_SLAVE, device_addr_7bit)
       i2c_smbus_write_byte_data(fd, reg_addr, data)
    """
    addr_write = (device_addr_7bit << 1) | 0   # R/W=0
    addr_read  = (device_addr_7bit << 1) | 1   # R/W=1
    if data is not None:
        frame = ["START", hex(addr_write), "ACK", hex(reg_addr), "ACK",
                 hex(data), "ACK", "STOP"]
        rw = "WRITE"
    else:
        frame = ["START", hex(addr_write), "ACK", hex(reg_addr), "ACK",
                 "RESTART", hex(addr_read), "ACK", "DATA", "NACK", "STOP"]
        rw = "READ"
    return {"device_addr": hex(device_addr_7bit), "reg": hex(reg_addr),
            "rw": rw, "frame": frame,
            "c_device": "/dev/i2c-1",
            "c_call": f"i2c_smbus_{'write' if data else 'read'}_byte_data(fd, {hex(reg_addr)}, ...)"}


PROTOCOL_COMPARISON = {
    "UART": {"wires": 2, "max_speed": "1 Mbaud",  "topology": "point-to-point",
             "use": "debug console, GPS, Bluetooth AT commands"},
    "SPI":  {"wires": 4, "max_speed": "100 MHz",  "topology": "master-slave, daisy",
             "use": "ADC, DAC, flash memory, STEAM ADC config"},
    "I2C":  {"wires": 2, "max_speed": "3.4 MHz",  "topology": "multi-master bus (7-bit addr)",
             "use": "sensors, EEPROM, real-time clock on RPi"},
    "CAN":  {"wires": 2, "max_speed": "1 Mbaud",  "topology": "differential bus, 128 nodes",
             "use": "automotive ECU, industrial, fault-tolerant"},
    "PCIe": {"wires": "2 per lane", "max_speed": "128 GB/s (PCIe 5.0 x16)",
             "topology": "point-to-point lanes",
             "use": "GPU (CUDA), NVMe SSD, STEAM high-speed data to host"},
}


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  dgs/edge_ai_protocols.py -- Edge AI, RF, Nuclear, Protocols")
    print("=" * 65)

    print("\n--- C File I/O (open a file) ---")
    print("  POSIX:  fd = open('data.bin', O_RDWR|O_CREAT, 0644)")
    print("  stdio:  fp = fopen('output.csv', 'w')")
    print("  SPI:    fd = open('/dev/spidev0.0', O_RDWR)  <- same call!")
    print("  I2C:    fd = open('/dev/i2c-1', O_RDWR)      <- same call!")
    print("  KEY: in POSIX, every device is a file. open() is universal.")
    print()
    print(C_ADR_EXPLANATION.strip())

    print("\n--- Edge AI: INT8 quantization + MAC energy ---")
    r = int8_mac_energy(n_macs=50_000, freq_GHz=1.0)
    print(f"  GS network (50K MACs):  {r['E_inference_pJ']:.1f} pJ/inference")
    print(f"  Throughput:             {r['throughput_fps']:.0f} fps at 1 GHz")
    print(f"  {r['vs_fp32']}")
    print(f"  Jalali use:             {r['jalali_use']}")

    w = np.array([0.345, -0.127, 0.891, -0.456, 0.012])
    q = quantize_weights(w, bits=8)
    print(f"  Quantization: {w} -> {q['w_int8']}")
    print(f"  Max error: {q['max_quant_error']:.2e}  ({q['compression']})")

    print("\n--- Wireless link budget (5 GHz WiFi, 10m) ---")
    r = rf_link_budget(P_tx_dBm=20, G_tx_dBi=2, G_rx_dBi=2,
                       freq_GHz=5.0, distance_m=10)
    print(f"  P_rx = {r['P_rx_dBm']} dBm  (FSPL={r['FSPL_dB']} dB)")
    print(f"  Noise floor = {r['noise_floor_dBm']} dBm")
    print(f"  SNR = {r['SNR_rx_dB']} dB  margin={r['link_margin_dB']} dB  OK={r['link_ok']}")
    lna = lna_noise_figure(G_dB=20, NF_dB=1.5)
    print(f"  LNA: G={lna['G_dB']} dB, NF={lna['NF_dB']} dB, T_eff={lna['T_eff_K']} K")
    print(f"  Rule: {lna['rule']}")

    print("\n--- Nuclear/Radiation detection ---")
    gm = geiger_muller_circuit()
    print(f"  GM tube: tau_RC={gm['tau_RC_ns']} ns, f_3dB={gm['f_3dB_kHz']} kHz")
    print(f"  Max count rate: {gm['max_count_rate_Hz']} Hz (dead time {gm['dead_time_us']} us)")
    print(f"  {gm['sipm_connection']}")
    dose = radiation_dose(activity_Bq=1e6, distance_m=1.0, exposure_hours=1.0)
    print(f"  Co-60, 1MBq at 1m, 1hr: {dose['dose_mSv']:.4f} mSv")
    print(f"  ({dose['fraction_annual_limit']*100:.3f}% of NRC annual limit)")

    print("\n--- Communication protocols ---")
    u = uart_frame(0x41, baud=115200)
    print(f"  UART 'A'(0x41) at {u['baud']} baud: {u['n_bits']} bits, {u['frame_us']} us")
    print(f"  Bits: {u['bits']}")
    sp = spi_frame(0x42, [0x1A, 0x2B])
    print(f"  SPI: {sp['frame_bytes']}  Mode={sp['mode']}")
    ic = i2c_transaction(0x48, 0x00, data=0xAB)
    print(f"  I2C write: {ic['frame']}")
    print()
    print(f"  {'Protocol':8s} {'Wires':6s} {'Speed':12s} {'Use'}")
    print(f"  {'-'*55}")
    for proto, d in PROTOCOL_COMPARISON.items():
        print(f"  {proto:8s} {str(d['wires']):6s} {d['max_speed']:12s} {d['use'][:40]}")


if __name__ == "__main__":
    demo()
