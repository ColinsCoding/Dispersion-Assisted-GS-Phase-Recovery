/*
 * frqbot.c - a timed free-response drill bot in ANSI C (C89).
 *
 * Original practice questions (written for this repo, not taken from any exam or standard) in:
 * ANSI C, laser safety (Z136-style concepts in our own words), AP Biology, AP Chemistry,
 * AP Psychology, AP Economics, and CompTIA. Each question has a time limit (15 minutes by
 * default, like one AP free-response part); a late answer scores 0, as on the exam.
 *
 *   frqbot                         all questions, shuffled, 900 s each
 *   frqbot --subject econ --n 2    two economics questions
 *   frqbot --limit 60 --seed 7     1-minute limit, a different order
 *   frqbot --list                  print the bank and exit
 *
 * Numeric answers are graded with a relative tolerance; written answers by keywords
 * ("a|b" means either word counts). Build:
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror frqbot.c -o frqbot -lm
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_KEYS 4
#define LINE 512

typedef enum { NUM, KEY } Kind;

typedef struct {
    const char *subject;
    const char *prompt;
    Kind kind;
    double value, tol;           /* NUM: answer and relative tolerance */
    const char *keys[MAX_KEYS];  /* KEY: required ideas; "a|b" = either */
    int need;                    /* KEY: how many ideas earn credit */
    const char *explain;
} Question;

static const Question BANK[] = {
    {"c", "int a = 7, b = 2; what does printf(\"%d\", a / b * b + a % b) print?", NUM, 7, 0,
     {0}, 0, "a / b is integer division = 3; 3 * 2 = 6; a % b = 1; 6 + 1 = 7 (always a for b != 0)."},
    {"c", "Why does the body of if (x = 0) { ... } never run?", KEY, 0, 0,
     {"assign", "0|zero", "false"}, 2, "= assigns 0 to x, the expression's value is 0, and 0 is false in C. Use == to compare."},
    {"laser", "Eyewear is rated OD 4 at the laser's wavelength. What fraction of the light gets through?", NUM, 1e-4, 0.01,
     {0}, 0, "Transmission = 10^-OD = 10^-4."},
    {"laser", "The exposure is 50 W/cm^2 and the MPE is 2.5e-3 W/cm^2. What minimum OD is needed?", NUM, 4.30, 0.01,
     {0}, 0, "OD = log10(E / MPE) = log10(2e4) = 4.30."},
    {"bio", "16% of a population shows a recessive trait (Hardy-Weinberg). What fraction are carriers (2pq)?", NUM, 0.48, 0.01,
     {0}, 0, "q^2 = 0.16 -> q = 0.4, p = 0.6, 2pq = 0.48."},
    {"bio", "Explain why an enzyme stops working at very high temperature.", KEY, 0, 0,
     {"denatur|unfold", "shape", "active site", "bond"}, 2, "Heat breaks hydrogen and other weak bonds, the protein unfolds (denatures), and the active site loses the shape that fits the substrate."},
    {"chem", "What is the pH of 0.0010 M HCl (a strong acid)?", NUM, 3.0, 0.01,
     {0}, 0, "Strong acid: [H+] = 1.0e-3 M, pH = -log10(1.0e-3) = 3.00."},
    {"chem", "How many moles are in 36.0 g of water (molar mass 18.02 g/mol)?", NUM, 2.00, 0.01,
     {0}, 0, "n = m / M = 36.0 / 18.02 = 2.00 mol."},
    {"psych", "After Pavlov's training, the bell alone makes the dog salivate. Name the role the bell plays.", KEY, 0, 0,
     {"conditioned", "stimulus"}, 2, "The bell is the conditioned stimulus (CS); salivating to it is the conditioned response."},
    {"psych", "Define negative reinforcement and say what it does to a behavior.", KEY, 0, 0,
     {"remov|take away|taking away", "unpleasant|aversive", "increas|strengthen|more likely"}, 2, "Removing an unpleasant stimulus after a behavior, which INCREASES that behavior. (It is not punishment.)"},
    {"econ", "Demand Qd = 100 - 2P, supply Qs = 20 + 2P. Find the equilibrium price.", NUM, 20, 0.001,
     {0}, 0, "Set Qd = Qs: 100 - 2P = 20 + 2P -> 4P = 80 -> P = 20, Q = 60."},
    {"econ", "Qd = 100 - 2P, Qs = 20 + 2P, and a price ceiling is set at P = 15. How large is the shortage?", NUM, 20, 0.001,
     {0}, 0, "Qd = 100 - 30 = 70, Qs = 20 + 30 = 50; shortage = 70 - 50 = 20 units."},
    {"econ", "Demand increases while supply stays the same. What happens to the equilibrium price and quantity?", KEY, 0, 0,
     {"price", "quantity", "increase|rise|up|higher"}, 3, "Demand shifts right along a fixed supply curve: equilibrium price and quantity both rise."},
    {"comptia", "How many usable host addresses does a /26 IPv4 subnet have?", NUM, 62, 0.001,
     {0}, 0, "32 - 26 = 6 host bits: 2^6 = 64 addresses, minus network and broadcast = 62."},
    {"comptia", "What is the default TCP port for HTTPS?", NUM, 443, 0.001,
     {0}, 0, "HTTPS uses TCP 443 (HTTP is 80, SSH is 22)."},
    {"comptia", "Name the three parts of the CIA triad.", KEY, 0, 0,
     {"confidential", "integrity", "availab"}, 3, "Confidentiality, integrity, availability."},
};
#define N_BANK ((int)(sizeof BANK / sizeof BANK[0]))

static unsigned long lcg_state = 1UL;
static unsigned long lcg(void)
{
    lcg_state = (lcg_state * 1103515245UL + 12345UL) & 0x7fffffffUL;
    return lcg_state;
}

static void lower(char *s)
{
    for (; *s; s++)
        *s = (char)tolower((unsigned char)*s);
}

/* does the answer contain any of the '|'-separated alternatives in key? */
static int has_idea(const char *answer, const char *key)
{
    char buf[128], *tok;
    strncpy(buf, key, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    for (tok = strtok(buf, "|"); tok != NULL; tok = strtok(NULL, "|"))
        if (strstr(answer, tok) != NULL)
            return 1;
    return 0;
}

static int grade(const Question *q, char *answer, char *why, size_t why_len)
{
    if (q->kind == NUM) {
        char *end;
        double x = strtod(answer, &end);
        double err;
        if (end == answer) {
            sprintf(why, "no number found");
            return 0;
        }
        err = q->value != 0.0 ? fabs(x - q->value) / fabs(q->value) : fabs(x);
        sprintf(why, "you: %g, answer: %g", x, q->value);
        return err <= (q->tol > 0.0 ? q->tol : 1e-9);
    } else {
        int k, hits = 0;
        char missing[256] = "";
        lower(answer);
        for (k = 0; k < MAX_KEYS && q->keys[k] != NULL; k++) {
            if (has_idea(answer, q->keys[k]))
                hits++;
            else if (strlen(missing) + strlen(q->keys[k]) + 3 < sizeof missing) {
                strcat(missing, " [");
                strcat(missing, q->keys[k]);
                strcat(missing, "]");
            }
        }
        if (why_len > 0)
            sprintf(why, "%d of %d ideas (need %d); missing:%s", hits, k, q->need, hits == k ? " none" : missing);
        return hits >= q->need;
    }
}

int main(int argc, char **argv)
{
    const char *subject = NULL;
    double limit = 900.0;
    int n = N_BANK, order[N_BANK], i, j, count = 0, score = 0, asked = 0;
    char line[LINE], why[300];

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--subject") == 0 && i + 1 < argc)
            subject = argv[++i];
        else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc)
            limit = atof(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            lcg_state = (unsigned long)atol(argv[++i]);
        else if (strcmp(argv[i], "--list") == 0) {
            for (j = 0; j < N_BANK; j++)
                printf("%-8s %s\n", BANK[j].subject, BANK[j].prompt);
            return 0;
        } else {
            fprintf(stderr, "usage: %s [--subject c|laser|bio|chem|psych|econ|comptia] [--n N] [--limit SEC] [--seed K] [--list]\n", argv[0]);
            return 2;
        }
    }
    if (limit <= 0.0 || n <= 0) {
        fprintf(stderr, "--limit and --n must be positive\n");
        return 2;
    }
    for (i = 0; i < N_BANK; i++)
        if (subject == NULL || strcmp(BANK[i].subject, subject) == 0)
            order[count++] = i;
    if (count == 0) {
        fprintf(stderr, "no questions for subject %s\n", subject);
        return 2;
    }
    for (i = count - 1; i > 0; i--) {          /* Fisher-Yates shuffle */
        int t;
        j = (int)(lcg() % (unsigned long)(i + 1));
        t = order[i]; order[i] = order[j]; order[j] = t;
    }
    if (n > count)
        n = count;

    printf("FRQ drill: %d question(s), %.0f s each. Answer on one line.\n", n, limit);
    for (i = 0; i < n; i++) {
        const Question *q = &BANK[order[i]];
        time_t t0 = time(NULL);
        double used;
        int ok;
        printf("\n[%d/%d] (%s) %s\n> ", i + 1, n, q->subject, q->prompt);
        fflush(stdout);
        if (fgets(line, sizeof line, stdin) == NULL)
            break;
        line[strcspn(line, "\r\n")] = '\0';
        if ((unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            memmove(line, line + 3, strlen(line + 3) + 1);   /* drop a UTF-8 byte-order mark */
        used = difftime(time(NULL), t0);
        asked++;
        ok = grade(q, line, why, sizeof why);
        if (used > limit) {
            printf("TIME: %.0f s > %.0f s limit, scored 0 (%s)\n", used, limit, why);
            ok = 0;
        } else {
            printf("%s (%s, %.0f s)\n", ok ? "CORRECT" : "NOT YET", why, used);
        }
        printf("why: %s\n", q->explain);
        score += ok;
    }
    printf("\nscore=%d/%d\n", score, asked);
    return 0;
}
