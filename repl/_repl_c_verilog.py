"""
repl/_repl_c_verilog.py
C + Verilog + Makefile dense reference.  PDF-cheatsheet style.
Windows-first: MSVC cl.exe + nmake paths shown alongside gcc/make.
Bit manipulation, strings, structs, Verilog FSM, testbench pattern.
"""
print("=" * 65)
print("C + VERILOG + MAKEFILE  REFERENCE  (cheatsheet)")
print("=" * 65)
print()

# ============================================================
# 1. C BIT MANIPULATION  (the ";;" operator hint)
# ============================================================
print("""=== 1. C BIT MANIPULATION ===

  OPERATORS:
    &   AND       a & b
    |   OR        a | b
    ^   XOR       a ^ b
    ~   NOT       ~a
    <<  left shift   a << n  (multiply by 2^n)
    >>  right shift  a >> n  (divide by 2^n, arithmetic if signed)

  COMMON PATTERNS:
    Set bit n:        x |=  (1u << n)
    Clear bit n:      x &= ~(1u << n)
    Toggle bit n:     x ^=  (1u << n)
    Test bit n:       (x >> n) & 1
    Low n bits mask:  (1u << n) - 1
    Round up to pow2: n==0? 1 : 1u << (32 - __builtin_clz(n-1))
    Popcount:         __builtin_popcount(x)      // GCC/Clang
                      __popcnt(x)                // MSVC
    Count trailing 0: __builtin_ctz(x)
    Swap nibbles:     ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)

  BITFIELDS IN STRUCT:
    typedef struct {
        uint32_t D      : 16;   // dispersion parameter  [0..65535]
        uint32_t n_iter :  8;   // GS iterations         [0..255]
        uint32_t flags  :  4;   // option bits
        uint32_t _pad   :  4;   // pad to 32 bits
    } GSConfig;

    GSConfig cfg = {.D=5000, .n_iter=50, .flags=0b0011};
    // sizeof(GSConfig) == 4 bytes

  ENDIANNESS CHECK:
    uint32_t x = 0x01020304;
    uint8_t *p = (uint8_t*)&x;
    // Little-endian (x86): p[0]=0x04  p[1]=0x03
    // Big-endian   (net):  p[0]=0x01  p[1]=0x02

  PORTABLE BYTE SWAP:
    uint32_t bswap32(uint32_t x) {
        return ((x & 0xFF000000u) >> 24)
             | ((x & 0x00FF0000u) >>  8)
             | ((x & 0x0000FF00u) <<  8)
             | ((x & 0x000000FFu) << 24);
    }
""")

# demonstrate in Python (same logic, different syntax)
import struct, ctypes

x = 0xDEADBEEF
print(f"  x = 0x{x:08X}")
print(f"  Set bit 4:    0x{x | (1<<4):08X}")
print(f"  Clear bit 4:  0x{x & ~(1<<4) & 0xFFFFFFFF:08X}")
print(f"  Toggle bit 4: 0x{x ^ (1<<4):08X}")
print(f"  Bit 4 set?    {bool((x >> 4) & 1)}")
print(f"  Popcount:     {bin(x).count('1')}")
bswap = struct.unpack('>I', struct.pack('<I', x))[0]
print(f"  Byte swap:    0x{bswap:08X}")
print()

# ============================================================
# 2. C STRINGS + MEMORY (the string/symbol reference)
# ============================================================
print("""=== 2. C STRINGS + MEMORY ===

  STRING LITERALS:
    char *s  = "hello";           // pointer to read-only data
    char  s[] = "hello";          // stack copy, mutable
    const char *s = "hello";      // correct for literals

  KEY FUNCTIONS (string.h):
    strlen(s)              // length, NOT including \\0
    strcpy(dst, src)       // UNSAFE: no bounds check
    strncpy(dst, src, n)   // safer, may not null-terminate
    snprintf(buf,n,fmt..)  // ALWAYS use this for building strings
    strcmp(a, b)           // 0 if equal, <0 if a<b, >0 if a>b
    strstr(haystack, needle)  // pointer to first match, NULL if not found
    strtol(s, &end, base)  // string -> long, base 10/16/2

  MEMORY:
    void *malloc(size_t n)          // allocate n bytes, uninitialized
    void *calloc(count, size)       // allocate + zero
    void *realloc(ptr, new_size)    // resize
    void  free(ptr)                 // must call exactly once

  COMMON MISTAKE:
    char buf[64];
    sprintf(buf, "%s", long_string);   // OVERFLOW if long_string > 63 chars
    snprintf(buf, sizeof(buf), "%s", long_string);  // CORRECT

  SAFE PATTERN FOR DYNAMIC STRINGS:
    size_t n = strlen(src) + 1;
    char *copy = malloc(n);
    if (!copy) { perror("malloc"); exit(1); }
    memcpy(copy, src, n);
    // ... use copy ...
    free(copy);

  NULL TERMINATION RULE:
    Every C string ends with '\\0' (char value 0).
    strlen counts chars BEFORE '\\0'.
    Buffer must be strlen(s) + 1 bytes.
""")

# ============================================================
# 3. C STRUCTS + TYPEDEF PATTERNS
# ============================================================
print("""=== 3. STRUCTS + POINTER PATTERNS ===

  STRUCT DEFINITION:
    typedef struct {
        float  *I1;       // intensity array 1
        float  *I2;       // intensity array 2
        int     N;        // signal length
        float   D1, D2;   // dispersion params
        int     n_iter;
    } GSData;

  ALLOCATE + FREE:
    GSData *gs = malloc(sizeof(GSData));
    gs->N  = 512;
    gs->I1 = malloc(gs->N * sizeof(float));
    gs->I2 = malloc(gs->N * sizeof(float));
    // ... use ...
    free(gs->I1); free(gs->I2); free(gs);

  POINTER ARITHMETIC:
    float *p = arr;
    p++;           // advance by sizeof(float) = 4 bytes
    p += 3;        // advance by 12 bytes
    *(p + i)       // same as arr[i] when p == arr

  FUNCTION POINTER (callback):
    typedef float (*KernelFn)(float r, void *params);
    float gaussian(float r, void *p) { return expf(-r*r); }
    KernelFn fn = gaussian;
    float v = fn(1.5f, NULL);

  CONST CORRECTNESS:
    void process(const float *in, float *out, int n);
    // in  cannot be modified through this pointer
    // out can be modified
""")

# ============================================================
# 4. MAKEFILE REFERENCE
# ============================================================
print("""=== 4. MAKEFILE REFERENCE ===

  SYNTAX:
    target: dependencies
    [TAB]   command          # MUST be a real TAB, not spaces

  VARIABLES:
    CC      = gcc
    CFLAGS  = -O2 -Wall -Wextra -std=c11
    LDFLAGS = -lm
    SRC     = main.c gs_core.c fno.c
    OBJ     = $(SRC:.c=.o)     # replace .c with .o

  PATTERN RULES:
    %.o: %.c
            $(CC) $(CFLAGS) -c $< -o $@
    # $<  = first dependency (the .c file)
    # $@  = target name (the .o file)
    # $^  = all dependencies

  FULL MINIMAL MAKEFILE:
    CC      = gcc
    CFLAGS  = -O2 -Wall -std=c11
    LDFLAGS = -lm
    TARGET  = gs_demo
    SRC     = main.c gs_core.c
    OBJ     = $(SRC:.c=.o)

    all: $(TARGET)

    $(TARGET): $(OBJ)
            $(CC) $(OBJ) $(LDFLAGS) -o $@

    %.o: %.c
            $(CC) $(CFLAGS) -c $< -o $@

    clean:
            rm -f $(OBJ) $(TARGET)

    .PHONY: all clean

  WINDOWS (nmake) DIFFERENCES:
    - Use backslash in paths: src\\main.c
    - CC = cl   CFLAGS = /O2 /W4
    - Object extension: .obj not .o
    - Linker: link /OUT:gs_demo.exe $(OBJ)
    - Clean: del /Q *.obj gs_demo.exe
    - No .PHONY support (just name targets uniquely)

  CUDA ADDITION:
    NVCC    = nvcc
    CUFLAGS = -O2 -arch=sm_89     # RTX 4060 = sm_89
    %.o: %.cu
            $(NVCC) $(CUFLAGS) -c $< -o $@
""")

# ============================================================
# 5. VERILOG REFERENCE
# ============================================================
print("""=== 5. VERILOG QUICK REFERENCE ===

  MODULE SKELETON:
    module gs_control #(
        parameter N = 512,
        parameter ITER = 50
    ) (
        input  wire        clk,
        input  wire        rst_n,
        input  wire [15:0] D1,
        input  wire [15:0] D2,
        output reg  [31:0] phase_out,
        output reg         done
    );
        // internal signals
        reg [7:0] iter_cnt;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                iter_cnt <= 8'd0;
                done     <= 1'b0;
            end else begin
                if (iter_cnt < ITER) begin
                    iter_cnt <= iter_cnt + 1;
                    done     <= 1'b0;
                end else begin
                    done <= 1'b1;
                end
            end
        end
    endmodule

  DATA TYPES:
    wire        -- combinational connection
    reg         -- storage (holds value between clk edges)
    integer     -- 32-bit, for loop counters in simulation
    [7:0]       -- 8-bit vector, [MSB:LSB]
    signed [7:0]-- signed 8-bit

  OPERATORS:
    &, |, ^, ~       -- bitwise
    &&, ||, !        -- logical (1-bit result)
    ==, !=           -- equality (x/z-aware: use === for exact)
    ===, !==         -- case equality (matches x and z)
    {a, b}           -- concatenation
    {4{a}}           -- replication (4 copies of a)
    a ? b : c        -- ternary mux

  ALWAYS BLOCKS:
    always @(*)      -- combinational (any input change)
    always @(posedge clk)   -- sequential, clock edge
    always @(posedge clk or negedge rst_n)  -- async reset

  BLOCKING vs NON-BLOCKING:
    =   blocking:     execute in order (use in combinational)
    <=  non-blocking: schedule all, assign at end of time step (use in sequential)

  FSM PATTERN (3-always):
    // State register
    always @(posedge clk) state <= next_state;
    // Next-state logic
    always @(*) begin
        case (state)
            IDLE:    next_state = start ? RUNNING : IDLE;
            RUNNING: next_state = done_flag ? DONE : RUNNING;
            DONE:    next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    // Output logic
    always @(*) begin
        out_valid = (state == DONE);
    end

  TESTBENCH SKELETON:
    `timescale 1ns/1ps
    module gs_control_tb;
        reg clk = 0, rst_n = 0;
        reg [15:0] D1 = 16'd5000, D2 = 16'd8000;
        wire [31:0] phase_out;
        wire done;

        gs_control #(.N(512), .ITER(50)) dut (
            .clk(clk), .rst_n(rst_n),
            .D1(D1), .D2(D2),
            .phase_out(phase_out), .done(done)
        );

        always #5 clk = ~clk;    // 100MHz

        initial begin
            $dumpfile("gs_control.vcd");
            $dumpvars(0, gs_control_tb);
            #20 rst_n = 1;
            @(posedge done);
            $display("done at t=%0t  phase=%h", $time, phase_out);
            #100 $finish;
        end
    endmodule

  COMPILE + SIMULATE (Linux/WSL):
    iverilog -o sim gs_control.v gs_control_tb.v
    vvp sim
    gtkwave gs_control.vcd

  COMPILE (Windows, Icarus Verilog):
    iverilog.exe -o sim.vvp gs_control.v gs_control_tb.v
    vvp.exe sim.vvp

  SYNTHESIS (Xilinx/Intel):
    // Vivado: Add sources -> Run Synthesis -> Check utilization report
    // Quartus: Analysis & Synthesis -> Fitter -> TimeQuest
""")

# ============================================================
# 6. C ARGC/ARGV + GETOPT PATTERN  (from gs_cli_demo.c)
# ============================================================
print("""=== 6. C ARGV + OPTION PARSING ===

  int main(int argc, char *argv[]) {
      // argv[0] = program name
      // argv[1..argc-1] = arguments
      // argv[argc] = NULL (sentinel)

      for (int i = 1; i < argc; i++) {
          if (strcmp(argv[i], "--help") == 0) { print_help(); }
          else if (strcmp(argv[i], "-D") == 0 && i+1 < argc) {
              D = atoi(argv[++i]);
          }
      }
  }

  GETOPT (POSIX, not MSVC native):
    #include <getopt.h>
    int opt;
    while ((opt = getopt(argc, argv, "D:n:h")) != -1) {
        switch(opt) {
            case 'D': D = atoi(optarg); break;
            case 'n': n_iter = atoi(optarg); break;
            case 'h': print_help(); exit(0);
            default:  fprintf(stderr, "unknown option\\n"); exit(1);
        }
    }

  WINDOWS ALTERNATIVE (no getopt):
    Use custom parse_args() with strcmp as in gs_cli_demo.c
    Or link getopt from vcpkg: vcpkg install getopt-win32

  COMPILE + RUN:
    gcc -O2 -Wall gs_cli_demo.c -lm -o gs_cli_demo
    ./gs_cli_demo -D 5000 -n 50 --mode demo

    cl /O2 gs_cli_demo.c /link /OUT:gs_cli_demo.exe
    gs_cli_demo.exe -D 5000 -n 50 --mode demo
""")

# ============================================================
# 7. WINDOWS BUILD WORKFLOW SUMMARY
# ============================================================
print("""=== 7. WINDOWS BUILD WORKFLOW ===

  TOOLS TO INSTALL:
    1. MSVC:     winget install Microsoft.VisualStudio.2022.BuildTools
    2. GCC:      winget install MSYS2.MSYS2  -> pacman -S mingw-w64-gcc
    3. CMake:    winget install Kitware.CMake
    4. Icarus:   https://bleyer.org/icarus/  (Verilog sim)
    5. CUDA:     https://developer.nvidia.com/cuda-downloads
    6. Python:   winget install Python.Python.3.12

  PROJECT STRUCTURE:
    project/
      src/          C source files
      include/      headers (.h)
      rtl/          Verilog source (.v)
      tb/           testbenches (_tb.v)
      repl/         Python exploration scripts
      notebooks/    Jupyter
      Makefile      gcc build
      CMakeLists.txt  cross-platform
      .gitignore

  GIT WORKFLOW:
    git checkout -b feature/bessel-modes
    # edit code
    git add src/bessel.c
    git commit -m "Add cylindrical Bessel mode solver"
    git push origin feature/bessel-modes
    # open PR on GitHub

  SENDING ONLY AI DATA (sparse):
    // Instead of sending full float array N=512:
    // Find significant components in Fourier domain
    // Send (index, value) pairs where |F[k]| > threshold
    // Reconstruct with IFFT + zero-fill
    // Compression ratio: depends on sparsity
    // GS uses this implicitly: phase stored as N floats,
    //   but meaningful info is in ~N/4 Fourier coefficients
""")

print("=" * 65)
print("Run with: py -3.12 repl/_repl_c_verilog.py")
print("Print to PDF: in terminal -> right-click -> Save as PDF")
print("=" * 65)
