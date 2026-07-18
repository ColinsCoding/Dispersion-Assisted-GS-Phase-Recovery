"""Pytest bootstrap: put the project root on sys.path so the top-level packages
(physics, optics, photonics, ...) import without an editable install."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
