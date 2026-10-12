"""Scarlett Particle Tagging: simulation-first optical particle characterization.

physics        Rayleigh / Rayleigh-Debye / Mie scattering, SEALS map, Brownian motion
sensor         illumination -> photodiode -> TIA -> ADC (behavioural)
estimator      diameter, index, concentration and motion estimators (reference)
asic           fixed-point behavioural model of a PROPOSED ASIC datapath
telemetry      RF tag packet format and link budget (behavioural)
synthetic      synthetic-data generator
visualization  plots

All numerical outputs are synthetic. See README, "What this project does not claim".
"""
__version__ = "0.1.0"
