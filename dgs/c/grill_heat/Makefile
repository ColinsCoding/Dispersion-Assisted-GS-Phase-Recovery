CC       := gcc
CFLAGS   := -O2 -Wall -lm

SRC      := grill_heat_fd.c
BIN      := grill_heat_fd

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN) > grill_heat_fd_output.txt

clean:
	rm -f $(BIN) grill_heat_fd_output.txt

.PHONY: all run clean
