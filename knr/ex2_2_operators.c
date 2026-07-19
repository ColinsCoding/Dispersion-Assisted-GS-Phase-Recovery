/* K&R Chapter 2 — Types, Operators, Expressions
 * Covers: enumerations, arithmetic/relational/logical operators,
 *         precedence table, ALU bit-ops, flip-flop model,
 *         C-to-SQL mapping, famous C idioms, Exercise 2-2.
 *
 * Compile:  gcc -Wall -o ex2_2_operators ex2_2_operators.c
 */
#include <stdio.h>
#include <string.h>   /* strlen */
#include <ctype.h>    /* tolower */

/* ====================================================================
 * ENUMERATIONS
 * enum assigns integer constants starting at 0 (or explicit value).
 * Advantage over #define: the compiler can warn on missed cases.
 * ==================================================================== */
enum Weekday  { MON=1, TUE, WED, THU, FRI, SAT, SUN };   /* 1..7      */
enum Boolean  { FALSE=0, TRUE };
enum Gate     { AND_GATE, OR_GATE, XOR_GATE, NAND_GATE, NOR_GATE };
enum SQLType  { SQL_INT, SQL_VARCHAR, SQL_FLOAT, SQL_BOOL, SQL_NULL };

/* CompTIA A+ relevant: common hardware states as enum */
enum PowerState { S0_ON, S1_IDLE, S3_SLEEP, S4_HIBERNATE, S5_OFF };

/* ====================================================================
 * ALU OPERATIONS  (arithmetic + bitwise — the actual hardware ops)
 * A real ALU implements these with NAND/NOR gates at silicon level.
 * ==================================================================== */
void alu_demo(void)
{
    unsigned int a = 180u;  /* 0b10110100 */
    unsigned int b = 109u;  /* 0b01101101 */

    printf("\n=== ALU / Arithmetic Operators ===\n");
    printf("a         = %3u  (0x%02X)\n", a, a);
    printf("b         = %3u  (0x%02X)\n", b, b);
    printf("a + b     = %u\n",   a + b);
    printf("a - b     = %u\n",   a - b);
    printf("a * b     = %u\n",   a * b);
    printf("a / b     = %u  (integer division)\n", a / b);
    printf("a %% b    = %u  (remainder)\n", a % b);

    /* Bitwise — the ALU's fundamental ops */
    printf("\n--- Bitwise (ALU gate-level) ---\n");
    printf("a & b     = 0x%02X  (AND  -- both bits 1)\n",  a & b);
    printf("a | b     = 0x%02X  (OR   -- either bit 1)\n", a | b);
    printf("a ^ b     = 0x%02X  (XOR  -- exactly one)\n",  a ^ b);
    printf("~a        = 0x%02X  (NOT  -- invert)\n",       (~a) & 0xFF);
    printf("a << 2    = 0x%02X  (left shift  = *4)\n",     a << 2);
    printf("a >> 2    = 0x%02X  (right shift = /4)\n",     a >> 2);

    /* Famous C bit tricks */
    printf("\n--- Famous C bit idioms ---\n");
    printf("a & (a-1) = 0x%02X  (clear lowest set bit)\n",  a & (a-1));
    printf("a & (-a)  = 0x%02X  (isolate lowest set bit)\n",a & (-(int)a));
    printf("a ^ a     = %u     (XOR self = 0, used to zero reg)\n", a ^ a);
    /* swap without temp: x^=y; y^=x; x^=y; */
    unsigned x = 42, y = 99;
    x ^= y; y ^= x; x ^= y;
    printf("XOR swap  : x=%u y=%u (were 42,99)\n", x, y);
    /* power-of-2 check */
    unsigned n = 64;
    printf("%u is power of 2: %s\n", n, (n && !(n & (n-1))) ? "yes" : "no");
}

/* ====================================================================
 * FLIP-FLOP MODEL IN C
 * A D flip-flop: Q_next = D on clock rising edge.
 * SR latch:      invalid if S=R=1 (metastability).
 * ==================================================================== */
typedef struct { int Q; int Qbar; } SRLatch;

SRLatch sr_latch(SRLatch prev, int S, int R)
{
    SRLatch next = prev;
    if      (S && !R) { next.Q = 1; next.Qbar = 0; }   /* Set   */
    else if (!S && R) { next.Q = 0; next.Qbar = 1; }   /* Reset */
    else if (!S && !R){ /* Hold */ }
    else              { next.Q = -1; next.Qbar = -1; }  /* Invalid S=R=1 */
    return next;
}

int d_flipflop(int D, int clk, int Q_prev)
{
    return clk ? D : Q_prev;   /* latch D on rising clock */
}

void flipflop_demo(void)
{
    printf("\n=== Flip-Flop / Digital Logic ===\n");
    SRLatch ff = {0, 1};
    printf("SR latch initial: Q=%d Qbar=%d\n", ff.Q, ff.Qbar);
    ff = sr_latch(ff, 1, 0); printf("S=1,R=0 -> Q=%d\n", ff.Q);
    ff = sr_latch(ff, 0, 0); printf("S=0,R=0 -> Q=%d (hold)\n", ff.Q);
    ff = sr_latch(ff, 0, 1); printf("S=0,R=1 -> Q=%d\n", ff.Q);
    ff = sr_latch(ff, 1, 1); printf("S=1,R=1 -> Q=%d (INVALID)\n", ff.Q);

    int D_seq[] = {1, 1, 0, 1, 0};
    int clk[]   = {1, 0, 1, 1, 0};
    int Q = 0;
    printf("D flip-flop sequence: D clk -> Q\n");
    for (int i = 0; i < 5; i++) {
        Q = d_flipflop(D_seq[i], clk[i], Q);
        printf("  D=%d clk=%d -> Q=%d\n", D_seq[i], clk[i], Q);
    }
}

/* ====================================================================
 * RELATIONAL AND LOGICAL OPERATORS + PRECEDENCE (C order)
 *
 * Precedence (high → low):
 *  1.  () [] -> .          (postfix)
 *  2.  ! ~ ++ -- + - * &  (unary, right-to-left)
 *  3.  * / %               (multiplicative)
 *  4.  + -                 (additive)
 *  5.  << >>               (shift)
 *  6.  < <= > >=           (relational)
 *  7.  == !=               (equality)
 *  8.  &                   (bitwise AND)
 *  9.  ^                   (bitwise XOR)
 * 10.  |                   (bitwise OR)
 * 11.  &&                  (logical AND)
 * 12.  ||                  (logical OR)
 * 13.  ?:                  (ternary, right-to-left)
 * 14.  = += -= ...         (assignment, right-to-left)
 * 15.  ,                   (comma)
 *
 * SHORT-CIRCUIT: && and || stop evaluating as soon as result is known.
 * ==================================================================== */
void relational_logical_demo(void)
{
    printf("\n=== Relational & Logical Operators ===\n");
    int a = 5, b = 10, c = 5;

    /* Relational */
    printf("a=%d b=%d c=%d\n", a, b, c);
    printf("a < b     : %d\n", a < b);
    printf("a > b     : %d\n", a > b);
    printf("a <= c    : %d\n", a <= c);
    printf("a >= c    : %d\n", a >= c);
    printf("a == c    : %d\n", a == c);
    printf("a != b    : %d\n", a != b);

    /* Logical */
    printf("a<b && b>c: %d  (AND)\n", a < b && b > c);
    printf("a>b || b>c: %d  (OR)\n",  a > b || b > c);
    printf("!(a==b)   : %d  (NOT)\n", !(a == b));

    /* Precedence trap: & vs && */
    printf("\n--- Precedence traps ---\n");
    /* Intentional: show that & binds tighter than == */
    printf("5 & 3 == 1  -> %d  (& binds tighter: 5 & (3==1) = 5&0 = 0)\n",
           5 & (3 == 1));   /* parenthesised to be explicit */
    printf("(5&3) == 1  -> %d  (correct: 1==1)\n", (5&3) == 1);
    printf("2+3 * 4     -> %d  (* before +: 2+12=14)\n", 2+3*4);
    printf("(2+3) * 4   -> %d\n", (2+3)*4);
}

/* ====================================================================
 * C-TO-SQL MAPPING
 * C struct <-> SQL table row.  Famous C idiom: self-referential struct
 * for linked lists maps to SQL adjacency list (parent_id).
 * ==================================================================== */
typedef struct Record {
    int         id;
    char        name[64];
    float       value;
    int         parent_id;   /* adjacency list (tree in SQL) */
    struct Record *next;     /* linked list in C */
} Record;

void c_to_sql_demo(void)
{
    printf("\n=== C struct -> SQL table ===\n");
    printf("/* SQL DDL equivalent of Record struct */\n");
    printf("CREATE TABLE record (\n");
    printf("    id        INTEGER PRIMARY KEY,\n");
    printf("    name      VARCHAR(64) NOT NULL,\n");
    printf("    value     FLOAT,\n");
    printf("    parent_id INTEGER REFERENCES record(id)\n");
    printf(");\n");
    printf("-- Linked list traversal = SQL recursive CTE:\n");
    printf("WITH RECURSIVE tree AS (\n");
    printf("  SELECT * FROM record WHERE parent_id IS NULL\n");
    printf("  UNION ALL\n");
    printf("  SELECT r.* FROM record r JOIN tree t ON r.parent_id = t.id\n");
    printf(") SELECT * FROM tree;\n");

    /* Build a small linked list */
    Record r1 = {1, "root",   1.0f, 0, NULL};
    Record r2 = {2, "child1", 2.5f, 1, NULL};
    Record r3 = {3, "child2", 3.7f, 1, NULL};
    r1.next = &r2; r2.next = &r3;
    printf("\n-- C linked list traversal (= SQL SELECT):\n");
    for (Record *p = &r1; p; p = p->next)
        printf("  id=%d name=%-8s value=%.1f parent=%d\n",
               p->id, p->name, p->value, p->parent_id);
}

/* ====================================================================
 * FAMOUS C CODE IDIOMS (0..626 mapping)
 * K&R classics that every C programmer must know.
 * ==================================================================== */
/* 1. getchar/putchar loop — the original Unix pipeline */
/* 2. strlen */
int my_strlen(const char *s) { int n; for (n=0; *s; s++) n++; return n; }
/* 3. strcpy */
void my_strcpy(char *t, const char *s) { while ((*t++ = *s++)) ; }
/* 4. atoi */
int my_atoi(const char *s)
{
    int n = 0, sign = 1;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-') { sign = -1; s++; }
    else if (*s == '+') s++;
    for (; *s >= '0' && *s <= '9'; s++) n = n * 10 + (*s - '0');
    return sign * n;
}
/* 5. reverse a string in place */
void my_reverse(char *s)
{
    int i = 0, j = (int)strlen(s) - 1;
    for (; i < j; i++, j--) { char t = s[i]; s[i] = s[j]; s[j] = t; }
}
/* 6. itoa (int to string) */
void my_itoa(int n, char *s)
{
    int i = 0, sign = n;
    if (n < 0) n = -n;
    do { s[i++] = (char)(n % 10 + '0'); } while ((n /= 10) > 0);
    if (sign < 0) s[i++] = '-';
    s[i] = '\0';
    my_reverse(s);
}
/* 7. lower — tolower without ctype */
int my_lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 'a' - 'A' : c; }
/* 8. power */
double my_power(double base, int exp)
{
    double result = 1.0;
    for (int i = 0; i < (exp < 0 ? -exp : exp); i++) result *= base;
    return exp < 0 ? 1.0/result : result;
}

void famous_c_code_demo(void)
{
    printf("\n=== Famous C Code Idioms ===\n");
    printf("my_strlen(\"hello\") = %d\n", my_strlen("hello"));
    char buf[32]; my_strcpy(buf, "photonics"); printf("my_strcpy -> %s\n", buf);
    printf("my_atoi(\"-42\")    = %d\n", my_atoi("-42"));
    char rev[] = "dispersion"; my_reverse(rev); printf("my_reverse -> %s\n", rev);
    char num[16]; my_itoa(12345, num); printf("my_itoa(12345) -> %s\n", num);
    printf("my_lower('A')     = %c\n", my_lower('A'));
    printf("my_power(2,10)    = %.0f\n", my_power(2, 10));

    /* 0..626: iterate like K&R getline example */
    printf("\nFirst 10 of range 0..626: ");
    for (int i = 0; i <= 9; i++) printf("%d ", i);
    printf("... 626\n");
}

/* ====================================================================
 * EXERCISE 2-2
 * Original loop (from K&R getline, p.29):
 *
 *   for (i=0; i<lim-1 && (c=getchar()) != '\n' && c != EOF; ++i)
 *       s[i] = c;
 *
 * Rewrite WITHOUT && or ||.
 * Solution: move each condition into a separate if+break inside the loop.
 * This is semantically identical because && short-circuits left-to-right
 * and stops at the first false — exactly what break does.
 * ==================================================================== */
#define MAXLINE 80

int my_getline_original(char s[], int lim)
{
    int c, i;
    for (i = 0; i < lim-1 && (c = getchar()) != '\n' && c != EOF; ++i)
        s[i] = c;
    if (c == '\n') { s[i] = '\n'; ++i; }
    s[i] = '\0';
    return i;
}

/* Exercise 2-2: equivalent loop, no && or || */
int my_getline_ex2_2(char s[], int lim)
{
    int c = 0, i;
    for (i = 0; i < lim - 1; ++i) {   /* condition 1: bounds check */
        c = getchar();
        if (c == EOF) break;           /* condition 2: EOF */
        if (c == '\n') break;          /* condition 3: newline */
        s[i] = c;
    }
    if (c == '\n') { s[i] = '\n'; ++i; }
    s[i] = '\0';
    return i;
}

/* ====================================================================
 * MAIN
 * ==================================================================== */
int main(void)
{
    /* Enumerations demo */
    printf("=== Enumerations ===\n");
    enum Weekday today = WED;
    enum PowerState ps = S0_ON;
    enum Gate g = XOR_GATE;
    printf("Weekday WED = %d (Mon=1..Sun=7)\n", today);
    printf("PowerState S0_ON = %d (CompTIA A+ ACPI S-states)\n", ps);
    printf("Gate XOR = %d\n", g);

    /* For/switch on enum */
    const char *days[] = {"","Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
    printf("Today is %s\n", days[today]);

    alu_demo();
    flipflop_demo();
    relational_logical_demo();
    c_to_sql_demo();
    famous_c_code_demo();

    printf("\n=== Exercise 2-2: loop without && or || ===\n");
    printf("Original for-loop condition:\n");
    printf("  i<lim-1 && (c=getchar())!='\\n' && c!=EOF\n");
    printf("\nEquivalent rewrite:\n");
    printf("  for (i=0; i<lim-1; ++i) {\n");
    printf("      c = getchar();\n");
    printf("      if (c == EOF)  break;   /* was: && c!=EOF  */\n");
    printf("      if (c == '\\n') break;   /* was: && c!='\\n' */\n");
    printf("      s[i] = c;\n");
    printf("  }\n");
    printf("\nWhy equivalent: && short-circuits left-to-right;\n");
    printf("  break exits the loop body at the same point.\n");
    printf("  Condition 1 (bounds) stays as the for-header.\n");

    printf("\nType a line (or Ctrl+D for EOF) to test my_getline_ex2_2:\n");
    printf("> ");
    char line[MAXLINE];
    int len = my_getline_ex2_2(line, MAXLINE);
    printf("Read %d chars: [%s]\n", len, line);

    return 0;
}
