/* Exercise 1-20: detab -- replace tabs with the proper number of blanks
 * to space to the next tab stop. TABSTOP is a symbolic parameter (#define)
 * rather than a variable: tab stops are a fixed convention for a given
 * run of the program, not something that changes at runtime, so a
 * compile-time constant is the right knob (still easy to retune).
 */
#include <stdio.h>

#define TABSTOP 8

int main(void)
{
    int c, col;

    col = 0;
    while ((c = getchar()) != EOF) {
        if (c == '\t') {
            int spaces = TABSTOP - (col % TABSTOP);
            while (spaces-- > 0) {
                putchar(' ');
                col++;
            }
        } else if (c == '\n') {
            putchar(c);
            col = 0;
        } else {
            putchar(c);
            col++;
        }
    }
    return 0;
}
