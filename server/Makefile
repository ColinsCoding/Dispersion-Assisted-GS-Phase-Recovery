# server/Makefile  —  Graviton 3 target (aarch64-linux-gnu)
# Cross-compile on x86 Linux/WSL:  apt install gcc-aarch64-linux-gnu libfftw3-dev

CC      := aarch64-linux-gnu-gcc
CFLAGS  := -O3 -march=armv8.4-a+sve -std=c11 -Wall -Wextra -ffast-math
LDFLAGS := -lm -lfftw3f -lpthread

TARGET  := optical_rx
SRCS    := optical_rx.c gs_solver.c opa_control.c
OBJS    := $(SRCS:.c=.o)

.PHONY: all clean check-cc

all: check-cc $(TARGET)

check-cc:
	@which $(CC) >/dev/null 2>&1 || \
	  (echo "ERROR: $(CC) not found.  Install with:" && \
	   echo "  sudo apt install gcc-aarch64-linux-gnu" && \
	   echo "Then re-run: make server" && exit 1)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET) (aarch64 ELF — deploy to Graviton instance)"
	@file $(TARGET) 2>/dev/null || true

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
