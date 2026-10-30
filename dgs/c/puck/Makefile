# CSC 60 style build: separate compilation, warnings as errors, a test target.
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -Werror -O2
LDLIBS = -lm

all: puck_game puck_test

puck.o: puck.c puck.h
	$(CC) $(CFLAGS) -c puck.c

puck_game: puck_game.c puck.o
	$(CC) $(CFLAGS) puck_game.c puck.o -o $@ $(LDLIBS)

puck_test: puck_test.c puck.o
	$(CC) $(CFLAGS) puck_test.c puck.o -o $@ $(LDLIBS)

test: puck_test
	./puck_test

clean:
	rm -f *.o puck_game puck_test puck_game.exe puck_test.exe

.PHONY: all test clean
