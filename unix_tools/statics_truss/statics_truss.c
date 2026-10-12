/* statics_truss.c -- a real Unix-style C program for 2D truss statics
 * (method of joints), the same physics as dgs/truss.py's solve_truss()
 * and dgs/truss_symbolic.py's solve_truss_symbolic(), independently
 * implemented in C rather than ported, and cross-checked against them.
 *
 * UNIX PHILOSOPHY: does one thing (solves a statically determinate 2D
 * truss), reads a plain-text truss description from stdin OR a file
 * argument, writes results to stdout, and uses exit codes to signal
 * success/failure -- so it composes with pipes and redirection like any
 * other Unix filter:
 *   ./statics_truss < truss.txt
 *   ./statics_truss truss.txt
 *   cat truss.txt | ./statics_truss | tee results.txt
 *
 * INPUT FORMAT (whitespace-separated, comments start with # to end of line):
 *   n_nodes n_members
 *   x y fix_x fix_y load_x load_y      (repeated n_nodes times, node 0..n_nodes-1)
 *   node_a node_b                      (repeated n_members times, 0-indexed)
 *
 * fix_x/fix_y are 0 or 1 (1 = that DOF is a support reaction, matching
 * dgs.truss's supports convention: pin=(1,1), roller=(0,1)). Loads are
 * in the same force units the caller wants forces reported in (N here,
 * matching this repo's own Python examples).
 *
 * METHOD: builds the SAME 2N-equation system dgs.truss.solve_truss does,
 * "A f = -load", where the unknowns are the member axial forces
 * (tension positive) followed by the reaction components, and A's
 * columns are each member's direction cosines. Solved by Gaussian
 * elimination with partial pivoting (no external linear-algebra library
 * -- this is meant to be a self-contained, portable Unix tool, matching
 * dgs.truss's own "just linear algebra" framing, done here in C instead
 * of relying on NumPy).
 *
 * Requires the truss to be statically determinate: n_members +
 * n_reactions == 2*n_nodes, exactly like dgs.truss_symbolic requires --
 * checked explicitly, not assumed.
 *
 * Build:  gcc -O2 -o statics_truss statics_truss.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_NODES 64
#define MAX_MEMBERS 128
#define MAX_UNKNOWNS (2 * MAX_NODES)

typedef struct { double x, y; int fix_x, fix_y; double load_x, load_y; } Node;
typedef struct { int a, b; } Member;

static void die(const char *msg) {
    fprintf(stderr, "statics_truss: %s\n", msg);
    exit(1);
}

/* Read all of `in` into a malloc'd buffer with every '#'-to-end-of-line
 * comment blanked out (replaced with spaces, so line/column positions in
 * any future error messages would still line up) -- so the documented
 * "# comments" behavior is real, not just claimed. Caller must free the
 * returned buffer. */
static char *read_and_strip_comments(FILE *in) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) die("out of memory");
    int c;
    while ((c = fgetc(in)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            buf = realloc(buf, cap);
            if (!buf) die("out of memory");
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';

    int in_comment = 0;
    for (size_t i = 0; i < len; i++) {
        if (buf[i] == '#') in_comment = 1;
        if (buf[i] == '\n') in_comment = 0;
        if (in_comment) buf[i] = ' ';
    }
    return buf;
}

/* Gaussian elimination with partial pivoting: solves A*x = b in place.
 * A is n x n (row-major, flattened), b is length n. Returns 0 on success,
 * -1 if the system is singular (a mechanism: not enough members/reactions
 * to be statically determinate and stable). */
static int gaussian_solve(double *A, double *b, int n) {
    for (int col = 0; col < n; col++) {
        int pivot = col;
        double best = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(A[row * n + col]);
            if (v > best) { best = v; pivot = row; }
        }
        if (best < 1e-9) return -1;   /* singular: unstable geometry (a mechanism) */

        if (pivot != col) {
            for (int k = 0; k < n; k++) {
                double tmp = A[col * n + k];
                A[col * n + k] = A[pivot * n + k];
                A[pivot * n + k] = tmp;
            }
            double tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }

        for (int row = col + 1; row < n; row++) {
            double factor = A[row * n + col] / A[col * n + col];
            for (int k = col; k < n; k++) A[row * n + k] -= factor * A[col * n + k];
            b[row] -= factor * b[col];
        }
    }

    for (int row = n - 1; row >= 0; row--) {
        double sum = b[row];
        for (int k = row + 1; k < n; k++) sum -= A[row * n + k] * b[k];
        b[row] = sum / A[row * n + row];
    }
    return 0;
}

int main(int argc, char **argv) {
    FILE *in = stdin;
    if (argc > 1) {
        in = fopen(argv[1], "r");
        if (!in) die("cannot open input file");
    }

    Node nodes[MAX_NODES];
    Member members[MAX_MEMBERS];
    int n_nodes = 0, n_members = 0;

    char *text = read_and_strip_comments(in);
    if (in != stdin) fclose(in);
    const char *cursor = text;
    int consumed;

    if (sscanf(cursor, "%d %d%n", &n_nodes, &n_members, &consumed) != 2)
        die("expected 'n_nodes n_members' on the first line");
    cursor += consumed;
    if (n_nodes <= 0 || n_nodes > MAX_NODES) die("n_nodes out of range");
    if (n_members <= 0 || n_members > MAX_MEMBERS) die("n_members out of range");

    for (int i = 0; i < n_nodes; i++) {
        if (sscanf(cursor, "%lf %lf %d %d %lf %lf%n",
                   &nodes[i].x, &nodes[i].y, &nodes[i].fix_x, &nodes[i].fix_y,
                   &nodes[i].load_x, &nodes[i].load_y, &consumed) != 6) {
            die("malformed node line (expected: x y fix_x fix_y load_x load_y)");
        }
        cursor += consumed;
    }
    for (int m = 0; m < n_members; m++) {
        if (sscanf(cursor, "%d %d%n", &members[m].a, &members[m].b, &consumed) != 2)
            die("malformed member line (expected: node_a node_b)");
        cursor += consumed;
        if (members[m].a < 0 || members[m].a >= n_nodes || members[m].b < 0 || members[m].b >= n_nodes)
            die("member references an out-of-range node index");
    }
    free(text);

    /* count reaction DOFs and build the (node,axis) -> unknown-index map */
    int reaction_node[MAX_UNKNOWNS], reaction_axis[MAX_UNKNOWNS];
    int n_reactions = 0;
    for (int i = 0; i < n_nodes; i++) {
        if (nodes[i].fix_x) { reaction_node[n_reactions] = i; reaction_axis[n_reactions] = 0; n_reactions++; }
        if (nodes[i].fix_y) { reaction_node[n_reactions] = i; reaction_axis[n_reactions] = 1; n_reactions++; }
    }

    int n_unknowns = n_members + n_reactions;
    int n_eq = 2 * n_nodes;
    if (n_unknowns != n_eq) {
        fprintf(stderr, "statics_truss: not statically determinate (members=%d + reactions=%d = %d, need 2*n_nodes=%d)\n",
                n_members, n_reactions, n_unknowns, n_eq);
        return 2;
    }

    double *A = calloc((size_t)n_eq * n_unknowns, sizeof(double));
    double *b = calloc((size_t)n_eq, sizeof(double));
    if (!A || !b) die("out of memory");

    for (int m = 0; m < n_members; m++) {
        int a = members[m].a, c = members[m].b;
        double dx = nodes[c].x - nodes[a].x, dy = nodes[c].y - nodes[a].y;
        double len = sqrt(dx * dx + dy * dy);
        if (len < 1e-12) die("zero-length member");
        double ux = dx / len, uy = dy / len;
        /* tension in member (a,c) pulls node a TOWARD c, and node c TOWARD a */
        A[(2 * a) * n_unknowns + m]     += ux;
        A[(2 * a + 1) * n_unknowns + m] += uy;
        A[(2 * c) * n_unknowns + m]     -= ux;
        A[(2 * c + 1) * n_unknowns + m] -= uy;
    }
    for (int k = 0; k < n_reactions; k++) {
        int col = n_members + k;
        int row = 2 * reaction_node[k] + reaction_axis[k];
        A[row * n_unknowns + col] += 1.0;
    }
    for (int i = 0; i < n_nodes; i++) {
        b[2 * i]     -= nodes[i].load_x;
        b[2 * i + 1] -= nodes[i].load_y;
    }

    if (gaussian_solve(A, b, n_eq) != 0) {
        fprintf(stderr, "statics_truss: singular system -- unstable geometry (a mechanism)\n");
        free(A); free(b);
        return 3;
    }

    printf("member forces (tension positive):\n");
    for (int m = 0; m < n_members; m++) {
        double f = b[m];
        printf("  %d-%d: %+.4f  (%s)\n", members[m].a, members[m].b, f, f > 0 ? "tension" : "compression");
    }
    printf("reactions:\n");
    for (int i = 0; i < n_nodes; i++) {
        double rx = 0.0, ry = 0.0;
        for (int k = 0; k < n_reactions; k++) {
            if (reaction_node[k] == i) {
                if (reaction_axis[k] == 0) rx = b[n_members + k];
                else ry = b[n_members + k];
            }
        }
        if (nodes[i].fix_x || nodes[i].fix_y)
            printf("  node %d: Rx=%+.4f  Ry=%+.4f\n", i, rx, ry);
    }

    free(A); free(b);
    return 0;
}
