/* Exercise 1-21: entab -- replace strings of blanks by the minimum
 * number of tabs and blanks to achieve the same spacing, using the
 * same tab stops as detab.
 *
 * When either a tab or a single blank would suffice to reach a tab
 * stop, prefer the blank: a tab buys nothing over a blank in that
 * case (same column advance, same byte count) but loses information
 * (visually ambiguous width, editor-dependent), so there's no reason
 * to prefer it. A tab is only emitted when it covers 2+ columns of
 * blanks, where it actually shrinks the output.
 */
#include <stdio.h>

#define TABSTOP 8

int main(void)
{
    int c, col, nblanks;

    col = 0;
    nblanks = 0;
    while ((c = getchar()) != EOF) {
        if (c == ' ') {
            nblanks++;
            col++;
        } else {
            /* flush pending blanks before handling c */
            int start = col - nblanks;
            while (nblanks > 0) {
                int next_stop = start + TABSTOP - (start % TABSTOP);
                if (next_stop <= col) {
                    /* a tab reaches next_stop; only worth it if it
                       covers more than one blank */
                    if (next_stop - start > 1) {
                        putchar('\t');
                        nblanks -= (next_stop - start);
                        start = next_stop;
                    } else {
                        putchar(' ');
                        nblanks--;
                        start++;
                    }
                } else {
                    putchar(' ');
                    nblanks--;
                    start++;
                }
            }
            if (c == '\t') {
                int tcol = col;
                int spaces = TABSTOP - (tcol % TABSTOP);
                putchar('\t');
                col += spaces;
            } else if (c == '\n') {
                putchar(c);
                col = 0;
            } else {
                putchar(c);
                col++;
            }
        }
    }
    return 0;
}
