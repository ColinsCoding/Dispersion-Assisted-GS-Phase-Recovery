/* Exercise 1-22: fold long input lines into two or more shorter lines
 * after the last non-blank character before column n. If a line has
 * no blank/tab before column n, fold it hard at column n (no infinite
 * growth on pathological input, e.g. a single huge token).
 */
#include <stdio.h>
#include <string.h>

#define MAXLINE 1000
#define FOLDCOL 60

void fold(char line[], int len);

int main(void)
{
    char line[MAXLINE];
    int c, i;

    i = 0;
    while ((c = getchar()) != EOF) {
        if (i >= MAXLINE - 1 || c == '\n') {
            line[i] = '\0';
            fold(line, i);
            putchar('\n');
            i = 0;
        } else {
            line[i++] = c;
        }
    }
    if (i > 0) {
        line[i] = '\0';
        fold(line, i);
    }
    return 0;
}

/* fold: print line broken into chunks of at most FOLDCOL chars,
 * breaking at the last blank/tab before the limit when one exists. */
void fold(char line[], int len)
{
    int start = 0;

    while (len - start > FOLDCOL) {
        int limit = start + FOLDCOL;
        int brk = -1;
        int j;

        for (j = limit - 1; j > start; j--) {
            if (line[j] == ' ' || line[j] == '\t') {
                brk = j;
                break;
            }
        }
        if (brk == -1) {
            /* no blank to break at: hard fold at the column limit */
            brk = limit;
        }
        fwrite(line + start, 1, brk - start, stdout);
        putchar('\n');
        start = brk;
        while (start < len && (line[start] == ' ' || line[start] == '\t'))
            start++;
    }
    fwrite(line + start, 1, len - start, stdout);
}
