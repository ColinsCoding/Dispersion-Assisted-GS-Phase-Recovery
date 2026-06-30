/* Exercise 1-23: remove all comments from a C program, while leaving
 * quoted strings and character constants (and their escape sequences)
 * untouched, since /* and // inside a string literal aren't comments. */
#include <stdio.h>

enum state { CODE, IN_STRING, IN_CHAR, IN_LINE_COMMENT, IN_BLOCK_COMMENT };

int main(void)
{
    int c, prev;
    enum state st = CODE;

    prev = 0;
    while ((c = getchar()) != EOF) {
        switch (st) {
        case CODE:
            if (prev == '/' && c == '/') {
                st = IN_LINE_COMMENT;
                c = 0; /* don't let this '/' become next prev's pending slash */
            } else if (prev == '/' && c == '*') {
                st = IN_BLOCK_COMMENT;
                c = 0;
            } else {
                if (prev == '/')
                    putchar('/'); /* prior slash was not a comment opener */
                if (c == '"') {
                    st = IN_STRING;
                    putchar(c);
                } else if (c == '\'') {
                    st = IN_CHAR;
                    putchar(c);
                } else if (c != '/') {
                    putchar(c);
                }
            }
            break;

        case IN_STRING:
            putchar(c);
            if (c == '\\') {
                int esc = getchar();
                if (esc != EOF)
                    putchar(esc);
                c = 0; /* not a closing quote */
            } else if (c == '"') {
                st = CODE;
            }
            break;

        case IN_CHAR:
            putchar(c);
            if (c == '\\') {
                int esc = getchar();
                if (esc != EOF)
                    putchar(esc);
                c = 0;
            } else if (c == '\'') {
                st = CODE;
            }
            break;

        case IN_LINE_COMMENT:
            if (c == '\n') {
                putchar(c);
                st = CODE;
            }
            break;

        case IN_BLOCK_COMMENT:
            if (prev == '*' && c == '/') {
                st = CODE;
                c = 0; /* don't let closing '/' look like a comment opener */
            }
            break;
        }
        prev = c;
    }
    return 0;
}
