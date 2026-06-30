/* Exercise 1-24: check a C program for rudimentary syntax errors:
 * unmatched (), [], {}. Skips over comments, string literals, char
 * constants, and their escape sequences so brackets inside them
 * don't get counted. Not a full parser -- just balance + line
 * tracking, which is what the exercise asks for ("rudimentary").
 */
#include <stdio.h>

#define MAXDEPTH 200

enum state { CODE, IN_STRING, IN_CHAR, IN_LINE_COMMENT, IN_BLOCK_COMMENT };

struct frame { int ch; int line; };

int main(void)
{
    int c, prev, line;
    enum state st = CODE;
    struct frame stack[MAXDEPTH];
    int sp = 0;
    int errors = 0;

    prev = 0;
    line = 1;

    while ((c = getchar()) != EOF) {
        if (c == '\n')
            line++;

        switch (st) {
        case CODE:
            if (prev == '/' && c == '/') {
                st = IN_LINE_COMMENT;
                c = 0;
            } else if (prev == '/' && c == '*') {
                st = IN_BLOCK_COMMENT;
                c = 0;
            } else if (c == '"') {
                st = IN_STRING;
            } else if (c == '\'') {
                st = IN_CHAR;
            } else if (c == '(' || c == '[' || c == '{') {
                if (sp >= MAXDEPTH) {
                    fprintf(stderr, "line %d: nesting too deep\n", line);
                    errors++;
                } else {
                    stack[sp].ch = c;
                    stack[sp].line = line;
                    sp++;
                }
            } else if (c == ')' || c == ']' || c == '}') {
                int want = (c == ')') ? '(' : (c == ']') ? '[' : '{';
                if (sp == 0 || stack[sp - 1].ch != want) {
                    fprintf(stderr, "line %d: unmatched '%c'\n", line, c);
                    errors++;
                } else {
                    sp--;
                }
            }
            break;

        case IN_STRING:
            if (c == '\\') {
                int esc = getchar();
                if (esc == '\n')
                    line++;
                if (esc == EOF) { st = CODE; }
            } else if (c == '"') {
                st = CODE;
            } else if (c == '\n') {
                fprintf(stderr, "line %d: missing terminating \" character\n", line - 1);
                errors++;
                st = CODE;
            }
            break;

        case IN_CHAR:
            if (c == '\\') {
                int esc = getchar();
                if (esc == '\n')
                    line++;
                if (esc == EOF) { st = CODE; }
            } else if (c == '\'') {
                st = CODE;
            } else if (c == '\n') {
                fprintf(stderr, "line %d: missing terminating ' character\n", line - 1);
                errors++;
                st = CODE;
            }
            break;

        case IN_LINE_COMMENT:
            if (c == '\n')
                st = CODE;
            break;

        case IN_BLOCK_COMMENT:
            if (prev == '*' && c == '/')
                st = CODE;
            break;
        }
        prev = c;
    }

    if (st == IN_BLOCK_COMMENT) {
        fprintf(stderr, "unterminated /* comment at end of file\n");
        errors++;
    }
    while (sp > 0) {
        sp--;
        fprintf(stderr, "line %d: unmatched '%c'\n", stack[sp].line, stack[sp].ch);
        errors++;
    }

    if (errors == 0)
        printf("brackets balanced, no rudimentary syntax errors found\n");
    else
        printf("%d error(s) found\n", errors);

    return errors != 0;
}
