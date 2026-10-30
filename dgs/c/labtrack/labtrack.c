/*
 * labtrack.c - lab records in ANSI C (C89) on SQLite: laser-safety training, bench clearance,
 * journal club, standards by country, and an eyewear optical-density check.
 *
 *   labtrack DB exec FILE.sql                     run a SQL file (schema.sql, seed.sql)
 *   labtrack DB report YYYY-MM-DD                 standards, journal club, clearance, eyewear
 *   labtrack DB add-laser NAME NM WATTS MM CLASS  insert with a prepared statement
 *
 * Build (MinGW or Linux):
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror labtrack.c -o labtrack -lsqlite3 -lm
 *
 * The eyewear check uses the simplified visible-light MPE also in dgs/laser_eyewear.py
 * (400-700 nm, 0.25 s blink: 1.8e-3 t^0.75 / t W/cm^2) and the beam's exit irradiance
 * 4P / (pi d^2). It is an educational check, not a substitute for a Laser Safety Officer.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sqlite3.h>

#define BLINK_S 0.25
#define PI 3.14159265358979323846

static int die(sqlite3 *db, const char *what)
{
    fprintf(stderr, "error: %s: %s\n", what, db ? sqlite3_errmsg(db) : "no database");
    if (db)
        sqlite3_close(db);
    return 1;
}

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    char *buf;
    if (f == NULL)
        return NULL;
    fseek(f, 0L, SEEK_END);
    n = ftell(f);
    rewind(f);
    buf = (char *)malloc((size_t)n + 1);
    if (buf != NULL) {
        if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
            free(buf);
            buf = NULL;
        } else {
            buf[n] = '\0';
        }
    }
    fclose(f);
    return buf;
}

/* visible MPE (W/cm^2) for an exposure of t seconds; negative if outside 400-700 nm */
static double mpe_visible(double wavelength_nm, double t_s)
{
    if (wavelength_nm < 400.0 || wavelength_nm > 700.0)
        return -1.0;
    return 1.8e-3 * pow(t_s, 0.75) / t_s;
}

static int cmd_exec(sqlite3 *db, const char *path)
{
    char *sql = read_file(path), *err = NULL;
    if (sql == NULL) {
        fprintf(stderr, "error: cannot read %s\n", path);
        return 1;
    }
    if (sqlite3_exec(db, sql, NULL, NULL, &err) != SQLITE_OK) {
        fprintf(stderr, "error: %s: %s\n", path, err);
        sqlite3_free(err);
        free(sql);
        return 1;
    }
    free(sql);
    printf("ok: %s\n", path);
    return 0;
}

static int cmd_report(sqlite3 *db, const char *date)
{
    sqlite3_stmt *st;
    const char *q;

    printf("== Laser-safety standards by country ==\n");
    q = "SELECT country, document, body, url FROM standard ORDER BY country";
    if (sqlite3_prepare_v2(db, q, -1, &st, NULL) != SQLITE_OK)
        return die(db, "prepare standards");
    while (sqlite3_step(st) == SQLITE_ROW)
        printf("%-24s %-22s %-34s %s\n", sqlite3_column_text(st, 0), sqlite3_column_text(st, 1),
               sqlite3_column_text(st, 2), sqlite3_column_text(st, 3));
    sqlite3_finalize(st);

    printf("\n== Journal club from %s ==\n", date);
    q = "SELECT j.held_on, m.name, j.topic, j.title, j.doi_url, "
        "(SELECT COUNT(*) FROM attendance a WHERE a.session_id = j.id) "
        "FROM journal_club j JOIN member m ON m.id = j.presenter_id "
        "WHERE j.held_on >= ? ORDER BY j.held_on";
    if (sqlite3_prepare_v2(db, q, -1, &st, NULL) != SQLITE_OK)
        return die(db, "prepare journal club");
    sqlite3_bind_text(st, 1, date, -1, SQLITE_TRANSIENT);
    while (sqlite3_step(st) == SQLITE_ROW)
        printf("%s  %-16s [%s] %s\n            %s  (attended: %d)\n", sqlite3_column_text(st, 0),
               sqlite3_column_text(st, 1), sqlite3_column_text(st, 2), sqlite3_column_text(st, 3),
               sqlite3_column_text(st, 4), sqlite3_column_int(st, 5));
    sqlite3_finalize(st);

    printf("\n== Bench clearance on %s ==\n", date);
    q = "SELECT name, role, laser_safety_expires, site_expires, "
        "CASE WHEN laser_safety_expires >= ?1 AND site_expires >= ?1 THEN 'CLEARED' "
        "     WHEN laser_safety_expires IS NULL OR laser_safety_expires < ?1 THEN 'NOT CLEARED: laser safety training' "
        "     ELSE 'NOT CLEARED: site-specific training' END "
        "FROM clearance ORDER BY member_id";
    if (sqlite3_prepare_v2(db, q, -1, &st, NULL) != SQLITE_OK)
        return die(db, "prepare clearance");
    sqlite3_bind_text(st, 1, date, -1, SQLITE_TRANSIENT);
    while (sqlite3_step(st) == SQLITE_ROW)
        printf("%-18s %-8s laser=%-10s site=%-10s %s\n", sqlite3_column_text(st, 0),
               sqlite3_column_text(st, 1),
               sqlite3_column_text(st, 2) ? (const char *)sqlite3_column_text(st, 2) : "-",
               sqlite3_column_text(st, 3) ? (const char *)sqlite3_column_text(st, 3) : "-",
               sqlite3_column_text(st, 4));
    sqlite3_finalize(st);

    printf("\n== Eyewear check (simplified visible MPE, %.2f s) ==\n", BLINK_S);
    q = "SELECT e.label, l.name, l.wavelength_nm, l.power_w, l.beam_diameter_mm, e.od, e.lo_nm, e.hi_nm "
        "FROM eyewear e JOIN laser l ON l.id = e.laser_id ORDER BY e.id";
    if (sqlite3_prepare_v2(db, q, -1, &st, NULL) != SQLITE_OK)
        return die(db, "prepare eyewear");
    while (sqlite3_step(st) == SQLITE_ROW) {
        double nm = sqlite3_column_double(st, 2), P = sqlite3_column_double(st, 3);
        double d_cm = sqlite3_column_double(st, 4) / 10.0, od = sqlite3_column_double(st, 5);
        double lo = sqlite3_column_double(st, 6), hi = sqlite3_column_double(st, 7);
        double mpe = mpe_visible(nm, BLINK_S), E = 4.0 * P / (PI * d_cm * d_cm), need;
        const char *verdict;
        if (nm < lo || nm > hi) {
            need = -1.0;
            verdict = "FAIL: laser wavelength outside the eyewear's rated band";
        } else if (mpe < 0.0) {
            need = -1.0;
            verdict = "UNCHECKED: outside the simplified 400-700 nm MPE; ask the LSO";
        } else {
            need = E > mpe ? log10(E / mpe) : 0.0;
            verdict = od >= need ? "OK" : "FAIL: optical density too low";
        }
        if (need < 0.0)
            printf("%-30s on %-28s E=%.4g W/cm2  need OD   n/a  have %.1f  %s\n",
                   sqlite3_column_text(st, 0), sqlite3_column_text(st, 1), E, od, verdict);
        else
            printf("%-30s on %-28s E=%.4g W/cm2  need OD %5.2f  have %.1f  %s\n",
                   sqlite3_column_text(st, 0), sqlite3_column_text(st, 1), E, need, od, verdict);
    }
    sqlite3_finalize(st);
    return 0;
}

static int cmd_add_laser(sqlite3 *db, char **a)
{
    sqlite3_stmt *st;
    const char *q = "INSERT INTO laser (name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) "
                    "VALUES (?, ?, ?, ?, ?)";
    int rc;
    if (sqlite3_prepare_v2(db, q, -1, &st, NULL) != SQLITE_OK)
        return die(db, "prepare insert");
    /* bound parameters: the text is never pasted into the SQL, so no SQL injection */
    sqlite3_bind_text(st, 1, a[0], -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(st, 2, atof(a[1]));
    sqlite3_bind_double(st, 3, atof(a[2]));
    sqlite3_bind_double(st, 4, atof(a[3]));
    sqlite3_bind_text(st, 5, a[4], -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(st);
    sqlite3_finalize(st);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "rejected: %s\n", sqlite3_errmsg(db));
        return 3;
    }
    printf("added laser %s\n", a[0]);
    return 0;
}

int main(int argc, char **argv)
{
    sqlite3 *db = NULL;
    int rc;
    if (argc < 3) {
        fprintf(stderr, "usage: %s DB exec FILE | report YYYY-MM-DD | add-laser NAME NM WATTS MM CLASS\n", argv[0]);
        return 2;
    }
    if (sqlite3_open(argv[1], &db) != SQLITE_OK)
        return die(db, "open");
    sqlite3_exec(db, "PRAGMA foreign_keys = ON;", NULL, NULL, NULL);
    if (strcmp(argv[2], "exec") == 0 && argc == 4)
        rc = cmd_exec(db, argv[3]);
    else if (strcmp(argv[2], "report") == 0 && argc == 4)
        rc = cmd_report(db, argv[3]);
    else if (strcmp(argv[2], "add-laser") == 0 && argc == 8)
        rc = cmd_add_laser(db, argv + 3);
    else {
        fprintf(stderr, "unknown command or wrong number of arguments\n");
        rc = 2;
    }
    sqlite3_close(db);
    return rc;
}
