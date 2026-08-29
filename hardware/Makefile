# FFT8 -- one 8-point FFT, two implementations that must agree.
#   make c        build + run the C golden model (fft8.c)
#   make verilog  compile + simulate the Verilog RTL (fft8.v + testbench)
# TDM scheduler -- round-robin time-division multiplexing, same pattern.
#   make tdm_c        build + run the C golden model (tdm_scheduler.c)
#   make tdm_verilog  compile + simulate the Verilog RTL (tdm_scheduler.v + testbench)
#   make all      everything; make clean  remove build artifacts
CC      = gcc
IVERILOG = iverilog
VVP      = vvp

c: fft8.c
	$(CC) -O2 -Wall -o fft8_c fft8.c -lm
	./fft8_c

verilog: fft8.v fft8_tb.v
	$(IVERILOG) -g2012 -o fft8.vvp fft8.v fft8_tb.v
	$(VVP) fft8.vvp

tdm_c: tdm_scheduler.c
	$(CC) -O2 -Wall -o tdm_scheduler_c tdm_scheduler.c -lm
	./tdm_scheduler_c

tdm_verilog: tdm_scheduler.v tdm_scheduler_tb.v
	$(IVERILOG) -g2012 -o tdm_scheduler.vvp tdm_scheduler.v tdm_scheduler_tb.v
	$(VVP) tdm_scheduler.vvp

all: c verilog tdm_c tdm_verilog

clean:
	rm -f fft8_c fft8_c.exe fft8.vvp tdm_scheduler_c tdm_scheduler_c.exe tdm_scheduler.vvp

.PHONY: c verilog tdm_c tdm_verilog all clean
