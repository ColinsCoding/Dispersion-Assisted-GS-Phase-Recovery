"""Integers and pointers, for real, in torch/CUDA: a CUDA kernel's global
thread index computation and a torch tensor's memory-layout indexing are
BOTH literal pointer arithmetic -- not an analogy.

CUDA THREAD INDEXING (what every CUDA kernel computes first):
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T* element_ptr = base_ptr + idx;   // pointer arithmetic: base + offset*sizeof(T)
This is exactly flat_index_from_multi_index applied to a 1D grid of
thread blocks.

TENSOR MEMORY LAYOUT (what every torch tensor access computes):
  flat_offset = sum(index[i] * stride[i] for i in dims)
  element_address = data_ptr + flat_offset * itemsize
A tensor's .stride() IS the pointer-arithmetic coefficients; a
non-contiguous view (e.g. a transpose) changes the strides but the SAME
arithmetic still finds the right element -- verified here directly
against torch's own indexing, not just asserted.

INTEGER OVERFLOW (a real, documented GPU bug class): CUDA kernels
historically computed thread/element indices in 32-bit int. A tensor
with more than 2^31-1 (~2.1 billion) elements overflows that index type
-- a real failure mode for very large tensors, checked here the same
way dgs.c_type_precision checks integer overflow elsewhere in this repo.

py-3.12 (torch) for consistency with this repo's torch-module
convention, though none of this specifically requires a GPU -- the
pointer-arithmetic verification works identically on a CPU tensor.
"""

import numpy as np
import torch

INT32_MAX = 2**31 - 1


def flat_index_from_multi_index(indices, strides):
    """offset = sum(index[i] * stride[i]) -- the actual pointer-
    arithmetic formula every tensor library uses to locate an element
    from its multi-dimensional index."""
    if len(indices) != len(strides):
        raise ValueError("indices and strides must have the same length")
    if any(s < 0 for s in strides):
        raise ValueError("strides must be non-negative")
    return sum(i * s for i, s in zip(indices, strides))


def cuda_global_thread_index(block_idx, block_dim, thread_idx):
    """The real CUDA formula every kernel computes first:
    idx = blockIdx.x * blockDim.x + threadIdx.x."""
    if block_idx < 0 or block_dim <= 0 or thread_idx < 0:
        raise ValueError("block_idx, thread_idx must be non-negative; block_dim must be positive")
    if thread_idx >= block_dim:
        raise ValueError("thread_idx must be less than block_dim")
    return block_idx * block_dim + thread_idx


def verify_pointer_arithmetic_matches_tensor(tensor, multi_index):
    """Compute a tensor element's location via RAW pointer arithmetic
    (flat_index_from_multi_index on tensor.stride()) and confirm it
    matches what tensor[multi_index] actually returns -- a real,
    checkable proof that stride-based indexing IS pointer arithmetic,
    not just a suggestive analogy."""
    if len(multi_index) != tensor.dim():
        raise ValueError("multi_index length must match tensor.dim()")
    for i, dim_size in zip(multi_index, tensor.shape):
        if not (0 <= i < dim_size):
            raise ValueError(f"index {i} out of bounds for dimension size {dim_size}")
    strides = tensor.stride()
    flat_offset = flat_index_from_multi_index(multi_index, strides)
    # walk the tensor's underlying 1D storage directly at that flat offset
    flat_view = tensor.reshape(-1) if tensor.is_contiguous() else tensor.contiguous().reshape(-1)
    # for a non-contiguous tensor, the RAW flat_offset indexes the
    # UNDERLYING (contiguous) storage, not the logical reshape -- so
    # compare against the untyped storage directly for full rigor
    storage_value = tensor.as_strided((1,), (1,), storage_offset=flat_offset).item()
    expected_value = tensor[tuple(multi_index)].item()
    return {
        "flat_offset": flat_offset,
        "value_via_pointer_arithmetic": storage_value,
        "value_via_tensor_indexing": expected_value,
        "match": abs(storage_value - expected_value) < 1e-9,
    }


def would_int32_index_overflow(n_elements):
    """Would a CUDA kernel computing element indices in 32-bit int
    overflow for a tensor of this many elements? A real, documented bug
    class for very large tensors (> ~2.1 billion elements)."""
    if n_elements <= 0:
        raise ValueError("n_elements must be positive")
    return n_elements > INT32_MAX


if __name__ == "__main__":
    print("=== CUDA global thread index: literal pointer arithmetic ===\n")
    block_dim = 256   # threads per block, a real common CUDA choice
    for block_idx, thread_idx in [(0, 0), (0, 255), (1, 0), (4, 100)]:
        idx = cuda_global_thread_index(block_idx, block_dim, thread_idx)
        print(f"  blockIdx={block_idx}, blockDim={block_dim}, threadIdx={thread_idx}  "
              f"->  global index = {idx}")

    print("\n=== Tensor stride-based indexing IS pointer arithmetic (verified) ===\n")
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    print(f"contiguous tensor, shape={tuple(t.shape)}, strides={t.stride()}")
    for idx in [(0, 0), (1, 2), (3, 5)]:
        result = verify_pointer_arithmetic_matches_tensor(t, idx)
        print(f"  index {idx}: flat_offset={result['flat_offset']}, "
              f"pointer-arithmetic value={result['value_via_pointer_arithmetic']}, "
              f"tensor[{idx}]={result['value_via_tensor_indexing']}, "
              f"match={result['match']}")

    print("\n--- now a NON-contiguous view (transpose) -- strides change, formula still works ---\n")
    t_T = t.t()   # transpose: a view, not a copy -- strides are swapped, no data moved
    print(f"transposed view, shape={tuple(t_T.shape)}, strides={t_T.stride()}")
    for idx in [(0, 0), (2, 1), (5, 3)]:
        result = verify_pointer_arithmetic_matches_tensor(t_T, idx)
        print(f"  index {idx}: flat_offset={result['flat_offset']}, "
              f"pointer-arithmetic value={result['value_via_pointer_arithmetic']}, "
              f"tensor[{idx}]={result['value_via_tensor_indexing']}, "
              f"match={result['match']}")

    print("\n=== Integer overflow: a real CUDA indexing bug class ===\n")
    for n in [1_000_000, 2_000_000_000, 2_500_000_000]:
        overflow = would_int32_index_overflow(n)
        print(f"  {n:,} elements: int32 index overflow = {overflow}")
    print(f"  (int32 max = {INT32_MAX:,} -- a tensor bigger than this needs int64 indexing")
    print(f"   in the CUDA kernel, or it silently wraps around to a negative/wrong address)")
