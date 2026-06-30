"""SQLite experiment logger for dgs modules.

SQL (Structured Query Language) is how scientists and engineers store,
query, and retrieve structured data. For experimental physics:
  - Every run is a row: (timestamp, module, parameters, result, notes)
  - Query by parameter: SELECT * WHERE D_ps2 > 5000 AND error < 0.01
  - Aggregate: SELECT AVG(converged_iter) GROUP BY signal_type

This module wraps Python's built-in sqlite3 into an OOP ExperimentLogger
class. No external dependencies -- sqlite3 is part of the Python standard
library. Same pattern as ScienceProblemSolver but SQL-backed for scale.

WHY SQL OVER JSON:
  JSON file: fine for <1000 records, no queries
  SQLite:    millions of records, indexed queries, concurrent readers
  PostgreSQL / MySQL: multi-user, cloud, production lab data systems

SQL CHEAT SHEET (the 5 commands that cover 80% of use cases):
  CREATE TABLE  -- define a table (columns + types)
  INSERT INTO   -- add a row
  SELECT        -- retrieve rows (with WHERE, ORDER BY, LIMIT, GROUP BY)
  UPDATE        -- change existing rows
  DELETE        -- remove rows
"""
import sqlite3
import json
import os
import time
from datetime import datetime


# ── ExperimentLogger ─────────────────────────────────────────────────

class ExperimentLogger:
    """SQLite-backed log of physics / photonics experiment runs.

    Each run stores: module name, input parameters (JSON), scalar result,
    any notes, and a Unix timestamp.

    Usage
    -----
    log = ExperimentLogger("experiments.db")
    log.insert("gs_tsdft", {"D_ps2": -5000, "n_iter": 60}, result=0.0012)
    log.insert("beat_freq", {"f1": 440, "f2": 443}, result=3.0, notes="piano tuning")
    log.query("SELECT * FROM runs WHERE module='gs_tsdft'")
    log.summary()
    log.best_n("gs_tsdft", metric="result", n=3, minimize=True)
    """

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   REAL    NOT NULL,
        datetime_s  TEXT    NOT NULL,
        module      TEXT    NOT NULL,
        params_json TEXT    NOT NULL,
        result      REAL,
        notes       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_module ON runs (module);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON runs (timestamp);
    """

    def __init__(self, db_path=":memory:"):
        """Open (or create) a SQLite database at db_path.
        Use ':memory:' for an in-memory database (no file, lost on close).
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        for stmt in self.CREATE_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)
        self.conn.commit()

    def insert(self, module, params, result=None, notes=None):
        """Log one experiment run.

        Parameters
        ----------
        module : str   -- name of the dgs module / experiment
        params : dict  -- input parameters (serialized to JSON)
        result : float -- scalar metric (error, frequency, SNR, ...)
        notes : str    -- free-text annotation
        """
        ts = time.time()
        dt_s = datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.conn.execute(
            "INSERT INTO runs (timestamp, datetime_s, module, params_json, result, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, dt_s, module, json.dumps(params), result, notes),
        )
        self.conn.commit()
        return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def query(self, sql, params=()):
        """Run arbitrary SQL SELECT and return list of dicts."""
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def all_runs(self, module=None):
        """Return all runs, optionally filtered by module name."""
        if module:
            return self.query("SELECT * FROM runs WHERE module=? ORDER BY timestamp",
                              (module,))
        return self.query("SELECT * FROM runs ORDER BY timestamp")

    def best_n(self, module, metric="result", n=5, minimize=True):
        """Return the n runs with best (lowest or highest) metric value."""
        order = "ASC" if minimize else "DESC"
        return self.query(
            f"SELECT * FROM runs WHERE module=? AND {metric} IS NOT NULL "
            f"ORDER BY {metric} {order} LIMIT ?",
            (module, n),
        )

    def summary(self):
        """Aggregate statistics per module: count, mean result, min, max."""
        return self.query(
            "SELECT module, COUNT(*) as n_runs, "
            "AVG(result) as mean_result, MIN(result) as min_result, "
            "MAX(result) as max_result "
            "FROM runs GROUP BY module ORDER BY n_runs DESC"
        )

    def delete_run(self, run_id):
        """Delete a specific run by its integer id."""
        self.conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.conn.commit()

    def count(self, module=None):
        """Number of runs logged, optionally filtered by module."""
        if module:
            r = self.conn.execute("SELECT COUNT(*) FROM runs WHERE module=?",
                                  (module,)).fetchone()
        else:
            r = self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return r[0]

    def close(self):
        self.conn.close()

    def __repr__(self):
        return f"ExperimentLogger('{self.db_path}', {self.count()} runs)"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── SQL cheat sheet as a module-level constant ────────────────────────

SQL_CHEAT_SHEET = """
SQL CHEAT SHEET -- the 5 commands for experimental data
========================================================

1. CREATE TABLE
   CREATE TABLE experiments (
       id      INTEGER PRIMARY KEY,
       D_ps2   REAL,
       n_iter  INTEGER,
       error   REAL
   );

2. INSERT INTO
   INSERT INTO experiments (D_ps2, n_iter, error)
   VALUES (-5000, 60, 0.0012);

3. SELECT (the big one)
   SELECT * FROM experiments WHERE D_ps2 < -1000 ORDER BY error ASC LIMIT 10;
   SELECT AVG(error), MIN(error) FROM experiments GROUP BY n_iter;
   SELECT a.*, b.error FROM experiments a JOIN results b ON a.id = b.run_id;

4. UPDATE
   UPDATE experiments SET notes = 'converged' WHERE error < 0.001;

5. DELETE
   DELETE FROM experiments WHERE error > 1.0;

TYPES: INTEGER, REAL, TEXT, BLOB (binary data -- e.g., numpy array as bytes)
AGGREGATES: COUNT(*), AVG(col), SUM(col), MIN(col), MAX(col)
FILTERING: WHERE, AND, OR, NOT, IN, BETWEEN, LIKE
SORTING: ORDER BY col ASC/DESC
JOINING: JOIN, LEFT JOIN (combine tables by shared key)
"""


def demo_sql_queries(db=None):
    """Run a set of example queries on an in-memory database."""
    if db is None:
        db = ExperimentLogger(":memory:")
    # seed with synthetic GS runs
    import random
    rng = random.Random(42)
    for D in [-500, -1000, -2000, -5000, -10000]:
        for trial in range(5):
            n_iter = rng.randint(20, 80)
            error = max(0.0, 1.0 / abs(D) * rng.gauss(100, 20))
            db.insert("gs_tsdft", {"D_ps2": D, "n_iter": n_iter, "trial": trial},
                      result=error)

    print("=== Summary by module ===")
    for row in db.summary():
        print(f"  {row['module']}: {row['n_runs']} runs, "
              f"mean_err={row['mean_result']:.4f}")

    print("\n=== Best 3 GS runs (lowest error) ===")
    for row in db.best_n("gs_tsdft", minimize=True, n=3):
        params = json.loads(row["params_json"])
        print(f"  id={row['id']}, D={params['D_ps2']}, error={row['result']:.4f}")

    print("\n=== Raw SQL: runs with D < -4000 and error < 0.01 ===")
    results = db.query(
        "SELECT id, params_json, result FROM runs "
        "WHERE module='gs_tsdft' AND result < 0.01"
    )
    print(f"  {len(results)} runs match")
    return db


if __name__ == "__main__":
    print(SQL_CHEAT_SHEET)
    with ExperimentLogger(":memory:") as db:
        demo_sql_queries(db)
