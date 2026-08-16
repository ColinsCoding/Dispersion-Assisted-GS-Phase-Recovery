"""Test dgs/thz_griffiths_vocab.py: the glossary DataFrame is well-formed
and actually covers the symbols used in dgs/thz_waveguide_dispersion_relation.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from dgs.thz_griffiths_vocab import vocab_table, VOCAB

df = vocab_table()

# 1. returns a real, well-formed DataFrame
assert isinstance(df, pd.DataFrame)
assert list(df.columns) == ["symbol", "name", "meaning", "formula", "source"]
assert len(df) == len(VOCAB)
assert len(df) >= 10

# 2. no empty cells -- every row is actually filled in, not a stub
for col in df.columns:
    assert (df[col].str.len() > 0).all(), f"empty entry found in column {col!r}"

# 3. no duplicate symbols
assert df["symbol"].duplicated().sum() == 0

# 4. covers the core symbols this glossary exists to explain
symbols_joined = " ".join(df["symbol"])
for expected in ["omega", "beta_2", "m_eff", "v_phase", "v_group", "k_c"]:
    assert expected in symbols_joined, f"expected {expected!r} to appear in the glossary"

print("all dgs.thz_griffiths_vocab tests passed")
