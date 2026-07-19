"""Test CUDA thread indexing and tensor stride-based pointer arithmetic:
the flat-offset formula, the real CUDA global-index formula, and cross-
checking that raw pointer arithmetic on a tensor's strides matches
torch's own indexing -- on both contiguous AND non-contiguous
(transposed) views. Also int32 overflow, a real CUDA indexing bug class.
Requires py-3.12 (torch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from dgs.torch import cuda_pointer_arithmetic as cpa

# 1. flat_index_from_multi_index: standard row-major offset formula
assert cpa.flat_index_from_multi_index((1, 2), (6, 1)) == 8
assert cpa.flat_index_from_multi_index((0, 0), (6, 1)) == 0
assert cpa.flat_index_from_multi_index((3, 5), (6, 1)) == 23

# 2. cuda_global_thread_index matches the real CUDA formula directly
assert cpa.cuda_global_thread_index(0, 256, 0) == 0
assert cpa.cuda_global_thread_index(0, 256, 255) == 255
assert cpa.cuda_global_thread_index(1, 256, 0) == 256
assert cpa.cuda_global_thread_index(4, 256, 100) == 4 * 256 + 100

# 3. verify_pointer_arithmetic_matches_tensor: contiguous tensor, several indices
t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
for idx in [(0, 0), (1, 2), (3, 5), (2, 3)]:
    result = cpa.verify_pointer_arithmetic_matches_tensor(t, idx)
    assert result["match"]
    assert result["value_via_pointer_arithmetic"] == result["value_via_tensor_indexing"]

# 4. SAME check on a non-contiguous (transposed) VIEW -- strides change,
#    but the raw pointer-arithmetic formula must still find the right
#    element (this is the real point: strides aren't cosmetic)
t_T = t.t()
assert not t_T.is_contiguous()
assert t_T.stride() != t.stride()
for idx in [(0, 0), (2, 1), (5, 3)]:
    result = cpa.verify_pointer_arithmetic_matches_tensor(t_T, idx)
    assert result["match"]

# 5. would_int32_index_overflow: correct boundary behavior around INT32_MAX
assert not cpa.would_int32_index_overflow(1_000_000)
assert not cpa.would_int32_index_overflow(cpa.INT32_MAX)
assert cpa.would_int32_index_overflow(cpa.INT32_MAX + 1)
assert cpa.would_int32_index_overflow(2_500_000_000)

# 6. input validation
for bad_call in [
    lambda: cpa.flat_index_from_multi_index((1, 2), (6,)),          # mismatched lengths
    lambda: cpa.flat_index_from_multi_index((1, 2), (-6, 1)),        # negative stride
    lambda: cpa.cuda_global_thread_index(-1, 256, 0),
    lambda: cpa.cuda_global_thread_index(0, 0, 0),
    lambda: cpa.cuda_global_thread_index(0, 256, -1),
    lambda: cpa.cuda_global_thread_index(0, 256, 256),               # thread_idx >= block_dim
    lambda: cpa.verify_pointer_arithmetic_matches_tensor(t, (0,)),   # wrong length
    lambda: cpa.verify_pointer_arithmetic_matches_tensor(t, (10, 0)),  # out of bounds
    lambda: cpa.would_int32_index_overflow(-1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.torch.cuda_pointer_arithmetic tests passed")
