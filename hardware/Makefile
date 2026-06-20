# FFT8 -- one 8-point FFT, two implementations that must agree.
#   make c        build + run the C golden model (fft8.c)
#   make verilog  compile + simulate the Verilog RTL (fft8.v + testbench)
#   make all      both; make clean  remove build artifacts
CC      = gcc
IVERILOG = iverilog
VVP      = vvp

c: fft8.c
	$(CC) -O2 -Wall -o fft8_c fft8.c -lm
	./fft8_c

verilog: fft8.v fft8_tb.v
	$(IVERILOG) -g2012 -o fft8.vvp fft8.v fft8_tb.v
	$(VVP) fft8.vvp

all: c verilog

clean:
	rm -f fft8_c fft8_c.exe fft8.vvp

.PHONY: c verilog all clean
