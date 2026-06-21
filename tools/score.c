/* score.c -- the "big logic": rank repo suggestions by priority.
 *
 * Reads candidate findings from stdin, one per line:
 *     <severity:int> <effort:int> <text...>
 * computes a priority score (impact-heavy, effort-light):
 *     score = severity^2 / effort
 * sorts descending, and writes ranked lines to stdout:
 *     <score>\t<text>
 *
 * This is the half a scanner is bad at -- numeric ranking of many items -- done
 * in C and fed by the JavaScript scanner over a pipe. Pure stdin/stdout, so it
 * composes with anything. Education.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXTEXT 1024
#define MAXITEMS 200000

typedef struct { double score; char text[MAXTEXT]; } Item;

static int cmp(const void *a, const void *b) {
    double d = ((const Item *)b)->score - ((const Item *)a)->score;
    return (d > 0) - (d < 0);
}

int main(void) {
    Item *items = malloc(sizeof(Item) * MAXITEMS);
    if (!items) { fprintf(stderr, "score: out of memory\n"); return 1; }
    int n = 0;
    char line[MAXTEXT + 64];

    while (n < MAXITEMS && fgets(line, sizeof line, stdin)) {
        char *p = line;
        long sev = strtol(p, &p, 10);
        long eff = strtol(p, &p, 10);
        if (eff <= 0) eff = 1;                 /* guard against divide-by-zero */
        while (*p == ' ' || *p == '\t') p++;   /* skip to the text */
        size_t L = strlen(p);
        while (L && (p[L - 1] == '\n' || p[L - 1] == '\r')) p[--L] = '\0';
        if (L == 0) continue;                  /* skip blank candidates */
        items[n].score = (double)sev * (double)sev / (double)eff;
        strncpy(items[n].text, p, MAXTEXT - 1);
        items[n].text[MAXTEXT - 1] = '\0';
        n++;
    }

    qsort(items, n, sizeof(Item), cmp);
    for (int i = 0; i < n; i++)
        printf("%.2f\t%s\n", items[i].score, items[i].text);

    free(items);
    return 0;
}
