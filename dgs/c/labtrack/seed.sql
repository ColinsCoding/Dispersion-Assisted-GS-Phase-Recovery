-- Synthetic example data. Names are made up; the papers are real (check each DOI before
-- presenting); the standards rows give the publishing body's home page, not the standard's text.

INSERT INTO member (id, name, role) VALUES
  (1, 'Avery (student)', 'student'),
  (2, 'Blake (grad)', 'grad'),
  (3, 'Casey (staff)', 'staff'),
  (4, 'Devon (student)', 'student');

INSERT INTO standard (country, body, document, scope, url) VALUES
  ('United States', 'ANSI / Laser Institute of America', 'ANSI Z136.1', 'safe use of lasers (MPE, NHZ, controls, training)', 'https://www.lia.org'),
  ('United States', 'ANSI', 'ANSI Z136 series (accreditation)', 'US national standards body', 'https://www.ansi.org'),
  ('International', 'IEC', 'IEC 60825-1', 'safety of laser products: classification and requirements', 'https://www.iec.ch'),
  ('European Union', 'CENELEC', 'EN 60825-1', 'European adoption of IEC 60825-1', 'https://www.cenelec.eu'),
  ('United Kingdom', 'BSI', 'BS EN 60825-1', 'UK adoption of IEC 60825-1', 'https://www.bsigroup.com'),
  ('Japan', 'JSA / JISC', 'JIS C 6802', 'Japanese laser product safety standard', 'https://www.jisc.go.jp'),
  ('Australia / New Zealand', 'Standards Australia', 'AS/NZS IEC 60825.1', 'adoption of IEC 60825-1', 'https://www.standards.org.au'),
  ('Canada', 'CSA Group', 'CSA E60825-1', 'Canadian adoption of IEC 60825-1', 'https://www.csagroup.org');

INSERT INTO training (member_id, topic, completed, expires) VALUES
  (1, 'laser safety', '2026-09-02', '2027-09-02'),
  (1, 'site-specific', '2026-09-10', '2027-09-10'),
  (2, 'laser safety', '2025-08-15', '2026-08-15'),   -- expired
  (2, 'laser safety', '2026-10-20', '2027-10-20'),   -- renewed
  (2, 'site-specific', '2026-10-21', '2027-10-21'),
  (3, 'laser safety', '2026-01-05', '2027-01-05'),   -- no site-specific yet
  (4, 'electrical', '2026-09-01', '2027-09-01');

INSERT INTO laser (id, name, wavelength_nm, power_w, beam_diameter_mm, hazard_class) VALUES
  (1, 'green alignment pointer', 532, 0.005, 1.0, '3R'),
  (2, 'HeNe 633 bench laser', 633, 0.020, 0.8, '3B'),
  (3, 'mode-locked fiber laser 1550', 1550, 0.050, 2.0, '3B');

INSERT INTO eyewear (label, laser_id, od, lo_nm, hi_nm) VALUES
  ('OD2 green goggles', 1, 2.0, 500, 560),
  ('OD2 green goggles (wrong use)', 2, 2.0, 500, 560),
  ('OD4 red goggles', 2, 4.0, 600, 700),
  ('OD5 IR goggles', 3, 5.0, 900, 1700);

INSERT INTO journal_club (id, held_on, presenter_id, title, doi_url, topic) VALUES
  (1, '2026-11-05', 2, 'Dispersive Fourier transformation for fast continuous single-shot measurements (Goda & Jalali, 2013)', 'https://doi.org/10.1038/nphoton.2013.131', 'photonics'),
  (2, '2026-11-12', 1, 'Phase retrieval algorithms: a comparison (Fienup, 1982)', 'https://doi.org/10.1364/AO.21.002758', 'phase retrieval'),
  (3, '2026-11-19', 3, 'Charge-insensitive qubit design derived from the Cooper pair box (Koch et al., 2007)', 'https://doi.org/10.1103/PhysRevA.76.042319', 'quantum information'),
  (4, '2026-12-03', 4, 'Simulating physics with computers (Feynman, 1982)', 'https://doi.org/10.1007/BF02650179', 'computing');

INSERT INTO attendance (session_id, member_id) VALUES (1, 1), (1, 2), (1, 3), (2, 1), (2, 2);
