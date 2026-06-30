CC      = gcc
CFLAGS  = -Wall -Wextra -std=c99 -O2

TARGETS = detab entab fold strip_comments syntax_check

.PHONY: all clean

all: $(TARGETS)

detab:         detab.c
entab:         entab.c
fold:          fold.c
strip_comments: strip_comments.c
syntax_check:  syntax_check.c

$(TARGETS):
	$(CC) $(CFLAGS) -o $@ $<

clean:
	-del /F /Q $(addsuffix .exe, $(TARGETS)) 2>nul
