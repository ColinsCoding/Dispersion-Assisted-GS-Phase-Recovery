"""Test photonics_manufacturing_funding_landscape() in dgs/sbir_portfolio.py
-- the real, hardware-specific funding mechanisms (AIM Photonics, MPW
shuttle runs, CHIPS Act, SBIR, private capital) distinct from the
existing pure-software/algorithm SBIR proposals in that file. Does not
re-test the pre-existing PROPOSALS/startup_phase_roadmap content, which
had no test coverage before this session and is out of scope here."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import sbir_portfolio as sbir

items = sbir.photonics_manufacturing_funding_landscape()

# 1. returns a non-empty list of well-formed entries
assert isinstance(items, list)
assert len(items) >= 4
for item in items:
    assert "mechanism" in item and "type" in item and "detail" in item
    assert isinstance(item["mechanism"], str) and len(item["mechanism"]) > 0
    assert isinstance(item["detail"], str) and len(item["detail"]) > 20

# 2. covers the specific real mechanisms this was built to cover
mechanisms = " ".join(item["mechanism"] for item in items)
assert "AIM Photonics" in mechanisms
assert "MPW" in mechanisms or "Multi-Project Wafer" in mechanisms
assert "CHIPS" in mechanisms
assert "SBIR" in mechanisms
assert "capital" in mechanisms.lower()

# 3. all entries are distinct mechanisms (no accidental duplicates)
names = [item["mechanism"] for item in items]
assert len(names) == len(set(names))

# 4. the SBIR entry explicitly cross-references this file's own proposals,
#    not a generic restatement
sbir_entries = [item for item in items if "SBIR" in item["mechanism"]]
assert len(sbir_entries) == 1
assert "$275K" in sbir_entries[0]["detail"] or "275K" in sbir_entries[0]["detail"]

print("all dgs.sbir_portfolio photonics-funding tests passed")
