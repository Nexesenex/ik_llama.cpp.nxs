# CUDA Bit-Exact Quantization for GGUF

## 1. Goal

Offload part of the `llama-quantize` tensor quantization work to the CUDA GPUs
while producing **byte-for-byte identical** GGUF tensor payloads to the CPU
reference quantizers. `Q8_0` is implemented and enabled with `--cuda-quantize`
(off by default); every other quant type in the target set is *provably
bit-exact* under the legacy (non-OLS) algorithms and can be added behind the
same entry.

This branch (`custom_pre_ols`) is scoped so that the GGUF-facing reference
quantizers do **not** use the OLS `d = sumqx/sumq2` scale refinement. OLS is
retained only for the KV-cache CUDA kernels (which never touch a GGUF), as
agreed.

## 2. Bit-exactness model

A quantized block is byte-for-byte portable across backends if and only if
every stored byte depends only on:

1. **max / min reductions** — exact and order-independent, so any parallel
   reduction (any thread count, any tree shape) produces the identical scalar;
2. **per-element rounding** — `t = f(x_i)` applied to a single value, with a
   fixed, backend-independent rounding mode (truncate / round-half-up /
   round-half-even / round-half-away);
3. **FP16 conversion** — `GGML_FP32_TO_FP16` is round-to-nearest-even
   (`_cvtss_sh` on MSVC, software nearest-even elsewhere); CUDA `__float2half_rn`
   matches bit-for-bit.

There must be **no cross-element floating-point accumulation** in the stored
values (no `sumqx/sumq2`, no `sums` used as scales). Under the legacy
algorithms this holds for the seven "simple block" quants.

## 3. Target set

### 3.1 Bit-exact legacy block quants

All block sizes are 32 values. `d`/`m` are derived only from max/min. The CPU
reference line numbers are from `ggml/src/ggml-quants.c` on `custom_pre_ols`.

| Type | CPU scale (`_ref`) | CPU quant idiom | Stored fields |
|------|--------------------|-----------------|---------------|
| `Q4_0` ✓ | `amax = max\|v\|; d = max/-8; id = d?1/d:0` (`:673`) | `(int8_t)(x*id + 8.5f)`, clamp `MIN(15,…)` | `d`(fp16), 16 nibbles `qs[j] = xi0 \| xi1<<4` |
| `Q4_1` | `min/max; d=(max-min)/15; m=min` (`:716`) | `(int8_t)((x-min)*id + 0.5f)`, clamp `MIN(15,…)` | `d,m`(fp16), nibbles |
| `Q5_0` | `amax; d=max/-16` (`:757`) | `(int8_t)(x*id + 16.5f)`, clamp `MIN(31,…)` | `d`(fp16), nibbles, `qh`(32b) |
| `Q5_1` | `min/max; d=(max-min)/31; m=min` (`:805`) | `(uint8_t)((x-min)*id + 0.5f)` (no clamp) | `d,m`(fp16), nibbles, `qh`(32b) |
| `Q6_0` | `amax; d=max/-32` (`:853`) | `(int8_t)(x*id + 32.5f)`, clamp `MIN(63,…)` | `d`(fp16), nibbles, `qh`(8B) |
Not implemented for now : | `Q6_1` | `min/max; d=(max-min)/63; m=min` (`:897`) | `(int)((x-min)*id + 0.5f)`, clamp `MIN(63,…)` | `d,m`(fp16), nibbles, `qh`(8B) |
| `Q8_0` ✓ | `amax; d=amax/127; id=d?1/d:0` (`:943`) | `roundf(x*id)` (round-half-away) | `d`(fp16), 32 int8 |

✓ = implemented on CUDA.

Round-trip rule: the CPU `(int8_t)/(int)(f)` cast **truncates toward zero**;
CUDA `(signed char)/(int)(f)` truncates identically. `roundf()` in both C and
CUDA is round-half-away-from-zero. FP16 is round-to-nearest-even on both sides.

### 3.2 Non-target types (stay CPU)

- `Q2_K/Q3_K/Q4_K/Q5_K/Q6_K`: scale selection uses `make_qkx2_quants` grid
  searches with float comparison tie-breaking tied to a fixed scan order; not
  provably backend-stable without replicating that order.
- `IQ1_S…IQ4_KT`: trellis/IQ search with `best_scale = sumqx/sumq2`
  (`ggml-quants.c` ~`13262+`). Same reason. Future work if exactness can be
  pinned to a single canonical scan order.
- `Q4_0` fast path: the public `quantize_row_q4_0` still dispatches to
  `iqk_quantize_q4_0` whose stored scale is `d = sumqx/sumq2`
  (`iqk_quantize.cpp:849`). Before `Q4_0` can join the CUDA set, the CPU
  dispatch must point to the legacy `quantize_row_q4_0_ref` (see §7).

## 4. Kernel design

### 4.1 Flat block tiling

The F32 tensor is a row-major buffer of `nrows * n_per_row` floats with
`n_per_row % 32 == 0`, so the per-row segmentation into 32-value quant blocks
equals the flat segmentation. A quant block is just `[ib*32, (ib+1)*32)` over
the flat buffer, `ib = 0 … nblocks-1`. No row bookkeeping is needed; this also
makes row splitting across multiple GPUs trivially safe.

### 4.2 Warp-per-quant-block

- `blockDim.x = 32` (one warp), `gridDim.x = nblocks`.
- **Phase A (scale):** lane `l` loads `x[ib*32 + l]`; warp-reduce
  `fmaxf(|x|, …)` via `__shfl_xor_sync` (exact, order-independent). All lanes
  recompute `d`/`id`/`m` from the reduced scalar with the exact CPU arithmetic
  (`max/-8`, `(max-min)/15`, …). Lane 0 stores `d`,`m` with `__float2half_rn`.
- **Phase B (quant):** each lane rounds its own element with the type's exact
  idiom. Nibble/high-bit packing is done after a `__syncthreads()` from a
  shared 32-byte `xi[]` so byte ownership is deterministic:
  - `Q4_0/Q4_1`: lane `j<16` builds `qs[j] = xi[j] | xi[j+16]<<4`.
  - `Q5_0/Q5_1`: same nibbles + `qh` bit `j` from `xi[j]`, bit `j+16` from
    `xi[j+16]` (32-bit little-endian write).
  - `Q6_0/Q6_1 (Q6_1 is not implemented)`: `qs[j]` as above; `qh[k] = hi2(xi[k]) | hi2(xi[k+16])<<2 |
    hi2(xi[k+8])<<4 | hi2(xi[k+24])<<6` with `hi2(e)=xi[e]>>4` (CPU `:887-888`).
  - `Q8_0`: lane `l` writes `qs[l] = (int8_t)roundf(x[ib*32+l]*id)` directly —
    no packing step.

### 4.3 Determinism

- max/min reduce: exact, any order → same scalar.
- division `1/d`, `max/-8`, `(max-min)/15`: IEEE-correctly-rounded scalar ops,
  identical on CPU and GPU.
- rounding and FP16: per-element, matched semantics (§2, §3.1).
- No cross-lane FP accumulation anywhere.

## 5. Integration into `llama-quantize`

Implemented as follows:

1. New CUDA translation unit `ggml/src/ggml-cuda/quantize_gguf.cu`
   (+ `.cuh`), auto-picked by the `ggml-cuda/*.cu` GLOB in
   `ggml/src/CMakeLists.txt`.
2. Public host entry exposed via `ggml-cuda.h` under `GGML_USE_CUDA`:
   `ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row)`.
   It owns a device context (device 0 for the POC), allocates device buffers,
   copies in, launches, copies out, frees, and checks every CUDA return code
   (returning `0` on failure so the caller aborts). Returns
   `nrows * ggml_row_size(Q8_0, n_per_row)`.
3. `llama_model_quantize_params` gains `bool cuda_quantize` (default false),
   set by the `--cuda-quantize` CLI flag in `examples/quantize/quantize.cpp`
   (ignored with a warning in non-CUDA builds).
4. `do_quantize()`: when `params->cuda_quantize` and the new type is eligible
   (`Q8_0` always; `Q4_0` only when there is no importance matrix and
   `--symmetric-q4-0` is off), routes each `ne[2]` (expert) slice through the
   CUDA entry and skips the CPU chunk loop. All other types (and non-eligible
   `Q4_0`) fall back to the CPU `ggml_quantize_chunk` path untouched.
5. `ggml_validate_row_data` still runs on the CUDA output (it must pass, since
   the bytes equal the CPU bytes).

## 6. Validation harness

`examples/unit_test_cuda` (target `unit_test_cuda`) verifies byte-for-byte
parity. It compares three producers of Q8_0 GGUF bytes on identical input:

1. **GPU:** `ggml_cuda_quantize_q8_0` (`ggml/src/ggml-cuda/quantize_gguf.cu`);
2. **CPU:** `ggml_quantize_chunk` — the fork's real llama-quantize path
   (`quantize_q8_0 -> quantize_row_q8_0_ref`);
3. **REF:** a local copy of `quantize_row_q8_0_ref` (`ggml-quants.c:943`) with
   the fp16 step done by `__float2half_rn`, no ggml internals.

`gpu vs cpu` answers "is the kernel byte-exact?", `cpu vs ref` answers "is the
fork's CPU path still the vanilla reference?", `gpu vs ref` ties the two
together with an independent baseline. The harness is spec-driven: a
`quant_spec` table lists each covered type (currently `Q8_0` and `Q4_0`) with
its `QK`, block size, CUDA entry, local `ref_*` copy and fill set, so adding a
type is one table entry. Fills are `random-uniform`, `weight-like` (90% of
32-value blocks tight around zero, 10% wide), and crafted `edge-cases`
(all-zero, single outlier, ±max, exact `.5` rounding ties, denormals,
huge/tiny magnitudes, mixed signs); a `> 1<<20`-block tensor exercises the CUDA
host wrapper's chunk loop. `test_slices` reproduces `do_quantize`'s `ne[2]`
slicing exactly: the CUDA and CPU branches both quantize one expert slice at a
time into consecutive slots, and both must equal a single whole-tensor call.

End-to-end: `llama-quantize` twice on a small F32 model (`Q8_0`), once CPU,
once `--cuda-quantize`; `memcmp` the resulting tensor payloads.

Build: `cmake --build build --target unit_test_cuda -j`. Run:
`unit_test_cuda [--seed N] [--device N] [--all-devices] [--big] [--huge] [--quick]`.

## 7. Known gaps / follow-ups

- ~~**`Q4_0` dispatch:** switch `quantize_row_q4_0` to the legacy
  `quantize_row_q4_0_ref`, then add the CUDA `Q4_0` kernel.~~ Done: the fork's
  CPU `quantize_row_q4_0` already dispatches to the legacy ref and the CUDA
  `Q4_0` kernel is in. `Q4_0` stays on the CPU only when an importance matrix is
  present or `--symmetric-q4-0` is requested.
- Add `Q4_1, Q5_0, Q5_1, Q6_0 (Q6_1 is not implemented)` kernels behind the same entry, one commit
  per type, each validated by the harness.
- Multi-GPU: rows are independent; later split row ranges across the 3 devices
  and add pinned-memory async upload. Does not affect bit-exactness.
- Reproducibility statement: CUDA kernels must never introduce an FMA or a
  reduction into a stored scale; the validation harness is the guard.
