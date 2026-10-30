-- labtrack schema: a lab's laser-safety training records, journal club, and laser/eyewear inventory.
-- Every physical or bookkeeping limit is a CHECK constraint ("bounds"), so bad data is rejected
-- by the database itself, not just by the program that writes it.
-- Educational tool: it does not replace your institution's Laser Safety Officer or ANSI Z136.1.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS member (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 80),
    role  TEXT NOT NULL CHECK (role IN ('student', 'grad', 'staff', 'faculty'))
);

-- Laser-safety standards by country or region: who publishes them and where.
CREATE TABLE IF NOT EXISTS standard (
    id       INTEGER PRIMARY KEY,
    country  TEXT NOT NULL,
    body     TEXT NOT NULL,
    document TEXT NOT NULL,
    scope    TEXT NOT NULL,
    url      TEXT NOT NULL CHECK (url LIKE 'https://%')
);

CREATE TABLE IF NOT EXISTS training (
    id        INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL REFERENCES member(id),
    topic     TEXT NOT NULL CHECK (topic IN ('laser safety', 'site-specific', 'chemical', 'electrical')),
    completed TEXT NOT NULL CHECK (date(completed) = completed),
    expires   TEXT NOT NULL CHECK (date(expires) = expires),
    CHECK (expires > completed)
);

CREATE TABLE IF NOT EXISTS laser (
    id                INTEGER PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    wavelength_nm     REAL NOT NULL CHECK (wavelength_nm BETWEEN 180 AND 1000000),
    power_w           REAL NOT NULL CHECK (power_w > 0 AND power_w <= 1000),
    beam_diameter_mm  REAL NOT NULL CHECK (beam_diameter_mm > 0 AND beam_diameter_mm <= 100),
    hazard_class      TEXT NOT NULL CHECK (hazard_class IN ('1', '1M', '2', '2M', '3R', '3B', '4'))
);

CREATE TABLE IF NOT EXISTS eyewear (
    id        INTEGER PRIMARY KEY,
    label     TEXT NOT NULL,
    laser_id  INTEGER NOT NULL REFERENCES laser(id),
    od        REAL NOT NULL CHECK (od BETWEEN 0 AND 10),
    lo_nm     REAL NOT NULL CHECK (lo_nm >= 180),
    hi_nm     REAL NOT NULL CHECK (hi_nm <= 1000000),
    CHECK (lo_nm < hi_nm)
);

CREATE TABLE IF NOT EXISTS journal_club (
    id           INTEGER PRIMARY KEY,
    held_on      TEXT NOT NULL CHECK (date(held_on) = held_on),
    presenter_id INTEGER NOT NULL REFERENCES member(id),
    title        TEXT NOT NULL,
    doi_url      TEXT NOT NULL CHECK (doi_url LIKE 'https://doi.org/10.%'),
    topic        TEXT NOT NULL CHECK (topic IN ('photonics', 'phase retrieval', 'quantum information',
                                                 'computing', 'laser safety'))
);

CREATE TABLE IF NOT EXISTS attendance (
    session_id INTEGER NOT NULL REFERENCES journal_club(id),
    member_id  INTEGER NOT NULL REFERENCES member(id),
    PRIMARY KEY (session_id, member_id)
);

-- Cleared for the bench on a given date: unexpired laser-safety AND site-specific training.
-- (How often training must be renewed is set by your institution, not by this schema.)
CREATE VIEW IF NOT EXISTS clearance AS
SELECT m.id AS member_id, m.name, m.role,
       MAX(CASE WHEN t.topic = 'laser safety'  THEN t.expires END) AS laser_safety_expires,
       MAX(CASE WHEN t.topic = 'site-specific' THEN t.expires END) AS site_expires
FROM member m LEFT JOIN training t ON t.member_id = m.id
GROUP BY m.id;
