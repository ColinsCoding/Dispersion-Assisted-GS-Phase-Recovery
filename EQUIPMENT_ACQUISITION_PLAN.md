# Equipment Acquisition Plan — Getting Real Measurements as a Solo Entrepreneur

**Status:** `REQUESTING_DATA.md` and `email_to_callen.txt` document the original plan
(ask Jalali Lab for I1/I2 traces from their own bench). That path is no longer being
pursued. This document replaces it with a self-funded path to real, non-synthetic
two-arm dispersive intensity measurements — the actual thing Task 4 asks for — without
depending on anyone else's lab.

**What "real measurements" requires, physically.** `dgs.gs_core.retrieve_phase` needs
two intensity-only time traces, I1(t) and I2(t), of the *same* optical (or RF) field
sent through two paths with different, known dispersion (D1 ≠ D2). Every tier below is
a different way to physically produce that pair of traces — the algorithm doesn't care
whether the dispersive element is a spool of fiber or a length of coax, only that D1 and
D2 differ and are known.

---

## Tier 1 — RF/coax bootstrap: prove it on real hardware for ~$1.5–3K, own it outright

This is the fastest, cheapest way to stop testing on synthetic data. `H(ν)=exp(iπDν²)`
is the same math whether the carrier is a 1550 nm photon or a 2.4 GHz RF tone — a length
of coaxial cable or a chirp filter is a dispersive delay line, and an envelope detector
(diode detector) gives intensity-only readout, exactly mirroring a photodiode. This
doesn't validate the *optical* link budget, but it validates the *receiver algorithm* on
genuinely captured hardware data — a real, defensible "we tested this on real
measurements" claim, not a synthetic one.

| Item | Role | Approx. cost (new) |
|---|---|---|
| USB oscilloscope/digitizer, 2-ch, ≥100 MSa/s (e.g. PicoScope 2000/3000 series, Digilent Analog Discovery 2) | Captures I1(t), I2(t) simultaneously | $300–$1,200 |
| RF signal generator or function generator with arbitrary/chirp capability | Generates the test waveform | $200–$800 (used bench units run cheaper) |
| 2× RF envelope/Schottky diode detector | Intensity-only readout (photodiode analog) | $50–$150 each |
| 2× coax cable spools, different lengths (e.g. 50 ft vs. 200 ft) or one length + one all-pass/chirp filter | The two "dispersive arms" — length difference sets D1 vs. D2 | $40–$150 |
| RF power splitter (2-way) | Splits the source into both arms | $20–$60 |
| Misc: SMA cables, attenuators, terminators | Bench plumbing | $50–$100 |
| **Tier 1 total** | | **≈ $1,500–$3,000** |

**First concrete deliverable:** a notebook (extending
`phase_retrieval_applications_and_trust.ipynb`'s framework) that runs
`retrieve_phase` on *actually captured* I1(t)/I2(t) from this bench, with the trust
checklist from that notebook applied to it before claiming success.

---

## Tier 2 — University/core-facility access: real telecom optics, no capital outlay

Before buying $30K+ of fiber-optic gear, most research universities (including
non-Jalali-Lab facilities) rent bench time on exactly this kind of equipment to outside
users, including solo entrepreneurs, on a fee-for-service basis — no faculty
relationship required, no data ownership fight, and no dependency on Yiming or Callen
specifically.

- **What to look for:** a photonics/optical-communications core facility or shared
  instrumentation lab. Search "[university name] photonics core facility external
  users" or "shared instrumentation facility fee schedule."
- **Examples of this model existing:** Boston University's Photonics Center runs shared
  lab facilities under Shared Laboratory Usage Agreements open to outside users; many
  university optics/engineering cores (e.g. CU Boulder's MIMIC facility, UW's Photonics
  Research Center) explicitly list "industrial" and "individual" fee-for-service access
  alongside academic use.
- **Typical structure:** hourly or daily bench-time rate (commonly in the low hundreds
  of dollars per day for a fiber-optics bench, though this varies by institution —
  confirm directly, don't assume) plus a standard external-user account/liability form.
- **What to bring:** the algorithm already works (`gs_verify.run_all()` passes,
  `ousd_critical_tech_capstone.ipynb` runs live) — you're paying for bench access to
  *capture* data, not to develop the receiver from scratch. A half-day to full-day
  session is realistically enough for one clean I1/I2 capture at a known D1/D2.
- **Action:** contact 2-3 nearby university core facilities directly (not just UCLA —
  broaden the search) and ask for their external-user fee schedule and equipment list
  before committing to Tier 3's capital purchase.

---

## Tier 3 — Own a minimal telecom-optics bench (after funding, or if usage justifies it)

The setup matching the actual Solli/Gupta/Jalali architecture and this repo's own
`D1_ps2=-600, D2_ps2=-1200` sample-data convention. Priced from real current listings
where found; flagged as an estimate where not.

| Item | Role | Approx. cost | Source |
|---|---|---|---|
| Fiber-coupled DFB laser, 1550 nm | Coherent source | $600–$2,800 (power-dependent) | market listings, e.g. laserdiodesource.com, civillaser.com |
| Modulator (or pulsed source instead of CW+modulator) | Generates the test signal | $1,000–$5,000+ | varies widely by type (EOM vs. direct-modulated) |
| 2× dispersion-compensating fiber (DCF) spool or chirped FBG, different D | The two dispersive arms | **not found priced publicly** — vendors (FiberMall, FS.com, Berkshire Photonics, Thorlabs) require a quote; budget-plan $500–$3,000/spool as an estimate, confirm directly | — |
| Benchtop EDFA (C-band, ~25–30 dB gain) | Compensates DCF's insertion loss | $3,700–$8,700 | Optilab listings |
| 2× high-speed InGaAs photodiode, telecom C-band, ≥10 GHz | Intensity-only readout | **not found priced publicly** — vendors (Agiltron, Edmund Optics, Koheron) require a quote; budget-plan $500–$2,500 each as an estimate | — |
| Real-time oscilloscope, ≥20 GHz analog bandwidth, ≥50 GSa/s, 2-ch | Captures both channels simultaneously — **the single largest cost item** | New: $50K–$150K+. Used/refurbished (eBay, TRS-RenTelco rental/sale, Keysight Used): commonly $10K–$40K depending on model/condition | eBay, TRS-RenTelco, Keysight Used |
| 50/50 fiber splitter, connectors, patch cables, isolators | Bench plumbing | $200–$500 | — |
| **Tier 3 total (used-equipment path)** | | **≈ $25,000–$60,000** | rough, dominated by the scope |
| **Tier 3 total (new-equipment path)** | | **≈ $80,000–$180,000+** | rough, dominated by the scope |

**Note on the biggest line item.** The real-time oscilloscope is almost always the
cost driver, not the optics — a strong reason to do Tier 2 (rent the scope's time)
before Tier 3 (own one outright), and to only buy once usage volume actually justifies
it.

---

## Funding path from Tier 1 to Tier 3

This repo already has an SBIR proposal framework (`dgs/sbir_portfolio.py`,
`dgs/ousd_alignment.py`) built around exactly this kind of dual-use photonics work —
worth pointing at directly rather than starting a funding search from scratch.

1. **SBIR/STTR Phase I, agency-dependent** — this repo's `dgs/sbir_portfolio.py` was
   flagged for its stale blanket "$275K" figure and has since been corrected per-agency
   (verified, not assumed): **NSF's own cap is $305,000** (NSF 26-510); **NIH, DOD, and
   most other agencies follow the SBA government-wide guideline, currently $314,363**
   (FY26 policy directive). Either is enough to fully fund a Tier 3 bench outright plus
   staff time, with room to spare.
2. **NSF I-Corps** — a smaller, faster ($25K-ish) award focused on customer discovery,
   often usable *before* a full Phase I application, and doesn't require lab equipment
   in hand yet — a plausible funding step between Tier 1 and Tier 3.
3. **University/regional accelerator or seed programs** — UCLA and other campuses
   (including, per `sample_data`'s own metadata, Sacramento State, which is apparently
   already adjacent to this exact problem) often run small non-dilutive seed grants or
   maker-space equipment funds for student/alumni ventures — worth checking before
   assuming SBIR is the only path.
4. **Sequencing that actually matters:** don't apply for Phase I funding citing only
   synthetic-data validation. Tier 1's real (if RF-domain) captured-data result, plus
   Tier 2's one real optical capture, is a substantially stronger Phase I technical
   narrative than simulation alone — do Tiers 1 and 2 first, then write the proposal.

---

## Concrete timeline: $1,500 to $305,000+

A realistic month-by-month sequence, not a wish list — each step is gated on the
previous one producing a real result, since that is what actually makes the next step
fundable.

| Timeframe | Action | Spend / inflow | Unlocks |
|---|---|---|---|
| Month 0 (this week) | Order the Tier 1 RF bench | −$1.5–3K (self-funded) | A real, capturable I1(t)/I2(t) pair within days, not months |
| Month 1 | Build the capture-and-verify notebook on Tier 1 data; apply the trust checklist before claiming anything | $0 | The first genuinely "tested on real hardware" claim for a pitch/proposal |
| Month 1–2 | Contact 2–3 university core facilities (fee-for-service, not Jalali-Lab-specific); confirm rate sheet and equipment list | $0 (inquiry) | Options for Tier 2 access, priced |
| Month 2–3 | Book one Tier 2 session (real telecom optics, no capital outlay) | −$200–1K (est. day-rate, confirm per facility) | One real optical-domain I1/I2 capture at known D1/D2 — the strongest technical evidence a Phase I reviewer will see |
| Month 3 | *(optional, parallel)* NSF I-Corps application | −time only, no cash outlay to apply | ~$25K non-dilutive if awarded, customer-discovery credibility for the Phase I narrative — does not require equipment in hand |
| Month 3–4 | Write and submit SBIR/STTR Phase I, citing the Tier 1 + Tier 2 real-data results directly (not synthetic-only) | −time (proposal writing) | Decision typically several months out, agency-dependent |
| Month ~8–12 (if awarded) | Phase I award lands | **+$305,000 (NSF) to +$314,363 (NIH/DOD/most agencies)** | Fully funds a Tier 3 bench outright (≈$25–60K used, ≈$80–180K new) *plus* real staff/founder time for the ~6-month Phase I period |
| Beyond $500K | Phase II, only after Phase I deliverables are met | up to **$1.75M** (this repo's own `sbir_portfolio.py` convention for Phase II scale) | Scaling past a single bench — outside this document's $1K–$500K scope, noted for completeness only |

**The honest gate in this timeline**: everything before the Phase I award is
self-funded and small (~$2–4K total cash outlay). The jump to $300K+ happens in ONE
step (the award), not gradually — which is exactly why Tiers 1–2 exist: they are cheap
enough to fund out of pocket, and their entire purpose is making that one big step
plausible rather than trying to shortcut to it on synthetic data alone.

---

## Immediate next steps

1. Order the Tier 1 bench (≈$1.5–3K) — this alone unblocks "real, non-synthetic
   measurements" this week, not after a grant decision.
2. Identify and email 2-3 university core facilities (fee-for-service, not
   Jalali-Lab-specific) about external-user access and their current equipment/rate
   sheet.
3. Build the Tier 1 capture-and-verify notebook once hardware arrives, applying
   `phase_retrieval_applications_and_trust.ipynb`'s trust checklist to the captured
   data before claiming anything about it.
4. Revisit NSF I-Corps / SBIR timing once Tier 1 (and ideally one Tier 2 session) has
   a real result to cite.

---

### Sources consulted for pricing (August 2026 web search, not a live feed — verify before purchasing)

- [TRS-RenTelco — Real-time Oscilloscopes 8GHz-20GHz](https://www.trsrentelco.com/products/oscilloscopes/realtime-oscilloscopes-8ghz-20ghz)
- [Keysight Used Equipment — Oscilloscopes](https://www.keysight.com/used/us/en/oscilloscopes)
- [Optilab — C-Band EDFA, Benchtop](https://www.optilab.com/products/erbium-doped-fiber-amplifier-c-band-benchtop)
- [LaserDiodeSource — 1550nm 20mW DFB Laser](https://shop.laserdiodesource.com/shop/1550nm-20mW-DFB-CoAxial-LasersCom)
- [CivilLaser — 1550nm fiber-coupled DFB laser modules](https://www.civillaser.com/index.php?main_page=product_info&products_id=3307)
- [Agiltron — High-Speed Fiber-Coupled InGaAs PIN Photodiode](https://agiltron.com/product/high-speed-fiber-coupled-ingaas-pin-photodiode/)
- [NSF SBIR — Budget guidance](https://seedfund.nsf.gov/how-to-submit/budget/)
- [BW&CO — NSF SBIR 2026 funding amounts](https://www.bwcoconsulting.com/fod/nsfsbir)
- [Boston University Photonics Center — Business Innovation Center / shared facilities](https://www.bu.edu/photonics/research/business-innovation-center/)
- [CU Boulder — Shared Research Instrumentation and Facilities](https://colorado.edu/engineering/research/facilities)
