# References

PDFs are not redistributed. Access via DOI or your institution's library.

Note (2026-08-18): `refs.db`'s `yao2022.pdf` entry had incorrect metadata
(listed as "Yao J, et al.", 2022) -- its actual full text is the Pu & Jalali
paper below (verified by reading it directly). Corrected in the database;
noted here so the mistake and the fix are both on the record.

**Foundational algorithm**

| Citation | DOI / Link |
|---|---|
| Gerchberg & Saxton, *Optik* 35(2):237–246, 1972 — original GS algorithm | [scinapse.io/papers/1484412996](https://www.scinapse.io/papers/1484412996) |

**The dispersive-Fourier-transform / STEAM lineage this repo is built on, in chronological order**

| Year | Citation | DOI / Link |
|---|---|---|
| 2009 | Solli, Gupta & Jalali, *Appl. Phys. Lett.* 95, 231108 — time-domain GS in the dispersive Fourier transform (the paper this repo's core algorithm implements) | [doi:10.1063/1.3271678](https://doi.org/10.1063/1.3271678) |
| 2009 | Goda, Tsia & Jalali, *Nature* 458, 1145–1149 — Serial Time-Encoded Amplified Microscopy (STEAM), the original demonstration | verify DOI before citing (not independently confirmed this session) |
| 2009 | Goda, Solli, Tsia & Jalali, *Phys. Rev. A* 80(4), 043821 — theory of amplified dispersive Fourier transformation | verify DOI before citing (not independently confirmed this session) |
| 2013/2014 | US Patent 8,870,060 B2 — Jalali, Goda & Tsia, "Apparatus and Method for Dispersive Fourier-Transform Imaging" (UC Regents; filed Feb 2013, priority to a Jul 2008 provisional, granted Oct 28 2014) — read directly this session, full breakdown in chat history | [Google Patents](https://patents.google.com/) — search `inventor:"Bahram Jalali"` |
| 2016 | Chen, Mahjoubfar, ... Jalali, *Sci. Rep.* 6, article 21471 — deep learning in label-free cell classification | verify DOI before citing (not independently confirmed this session) |
| 2017 | Mahjoubfar, Churkin, ... Jalali, *Nat. Photonics* 11(6), 341–351 — "Time stretch and its applications" (review) | verify DOI before citing (not independently confirmed this session) |
| 2019 | Li, Mahjoubfar, ... Jalali, *Sci. Rep.* 9, article 11088 — deep cytometry: real-time inference in cell sorting and flow cytometry | verify DOI before citing (not independently confirmed this session) |
| 2021 | Pu & Jalali, *Optics Express* 29(13), 20786 — neural network time-stretch spectral regression (read directly this session; PDF's actual content, not the mislabeled `yao2022.pdf` metadata) | [doi:10.1364/OE.426178](https://doi.org/10.1364/OE.426178) |

The 2016/2017/2019 rows are titles+journals already cited (with fuller
author lists) in `dgs/sbir_portfolio.py`'s P2/P3/P5 `verified_citations`
fields -- reproduced here in date order rather than re-verified from
scratch. DOIs marked "verify before citing" were not independently
looked up this session; don't cite them in a real proposal without
confirming first.

**Distinct from the above:** the 90-degree optical hybrid work
(`projects/vpi_hybrid90deg/`) is a VPIphotonics vendor-documented,
industry-standard component (Photonic Modules > Passive Components >
Hybrid90deg) used in coherent receiver design generally -- it is NOT
confirmed to be a Jalali patent or Jalali-lab-specific IP, and should not
be cited as one. It sits in the same coherent-detection technology space
as the papers above (see `notebooks/hybrid90deg_phase_retrieval_mie.ipynb`
for how the two connect), but that's a topical connection, not a
citation lineage.
