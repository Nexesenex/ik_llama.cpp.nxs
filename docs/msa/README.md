# MiniMax-M3 block-sparse attention (`--msa`)

MiniMax-M3 ships a lightning-indexer that scores blocks of the KV cache so attention can read only
the top-k blocks instead of the whole cache. This branch implements that path. It is **off by
default**; nothing changes for any other model, or for MiniMax-M3 without the flag.

The payoff is long context: attention cost stops growing linearly with `n_kv`. Below the crossover
it is a net loss, because the indexer has its own O(n_kv) cost. See *Where it wins and loses*.

## Using it

```
llama-cli   -m <MiniMax-M3 GGUF> --msa --msa-gather -c 65536 -t 48 -fa 1 -ngl 0
llama-server -m <MiniMax-M3 GGUF> --msa --msa-gather -c 65536 -t 48 -fa 1 -ngl 0 --jinja
```

- `--msa` enables sparse attention. `--msa-gather` additionally attends only the selected cells via
  an index list, rather than masking the full cache. Both default off.
- **`--jinja` is required for `llama-server`'s `/v1/chat/completions`.** Without it MiniMax-M3's
  chat template is rejected and every chat request returns **HTTP 500**
  (`this custom template is not supported, try using --jinja`). The `/completion` endpoint works
  without it. This is a MiniMax-M3 template requirement, not an MSA one, but it is the first thing
  a new tester hits.
- `--msa-min-kv N` runs dense while the cache is smaller than N and sparse at or above it, so you
  can avoid the below-crossover loss. The crossover is machine-specific AND phase-dependent — measure yours; on a dual
  Xeon 8260 decode's is around 23,000 and prefill's is lower. See *Where it wins and loses*.
- `--msa-gather` falls back to the mask path for tensor-parallel attention, `-fa 0`, an `n_kv` that
  is not a multiple of the block size, and any model whose head counts do not divide.

GGUFs converted by mainline llama.cpp are read directly. Mainline spells the indexer
hyper-parameters `attention.indexer.*` and its tensors `blk.N.indexer.{q,k}_proj`; our converter
wrote `attention.sparse_*` and `blk.N.index_{q,k}`. Both spellings are accepted.

**If the GGUF has no indexer tensors, `--msa` is accepted and logged but silently runs dense.**
Check for the indexer in your conversion if you see no change.

## Where it wins and loses

- **The crossover is phase-dependent, and quoting one number for both phases is wrong.** Decode
  crosses around 23,000 KV on this box (0.79x at 16k). Prefill crosses lower: ~7% slower at 4k-8k,
  but already ahead by 16.5k (43.67 dense vs 46.80 sparse t/s) and 1.42x by 33k. Those prefill
  figures are PRE-REBASE and have not been re-measured. The indexer scores every cached token on
  every decoded token; that cost is smaller than dense's but not zero, which is why the curves
  cross rather than the sparse path dominating everywhere.
- **`--msa-min-kv N` applies ONE threshold to BOTH phases.** Setting it at decode's crossover also
  forces prefill dense the whole way up to N, which cost ~15% prefill on a 33k prompt (also
  PRE-REBASE, same source as the prefill figures above). There is no
  separate prefill threshold on this branch.
- **No help on hybrid GPU.** With 8 of 61 layers on four P100s, dense reaches 53.37 t/s prefill and
  every MSA arm lands between 30 and 34. MSA nearly doubles backend graph splits (1410 vs 774) and
  every split is a synchronisation. Use dense there.
- **All measurement here is CPU-only.** The `-ngl > 0` path is implemented but has not been
  benchmarked post-rebase. Treat GPU numbers as unmeasured.

## Measurement discipline used here

Decode and prefill have very different noise on this box, and quoting one figure for both is wrong.
Three interleaved repeats per arm at n_kv 2,240, same binary, same session:

| arm | n | prefill spread (sd) | decode spread (sd) |
|---|---:|---:|---:|
| dense | 4 | **3.11%** | **0.41%** |
| gather | 5 | **4.01%** | **1.12%** |

So a decode difference above ~3% is real; a prefill difference below ~8% is not. Prefill scatters
about **7.5x** more than decode, close to `sqrt(192 decode tokens / 4 prefill ubatches) = 6.9`.
A run taken immediately after a change of regime came in 18% low, so first-run-after-a-change is
discarded.

## Results

### Re-measured on the current base (2026-08-27/28)

This branch was rebased from a 2026-01-10 base onto `ikawrakow/main` `7cff686d`, which **voided every
earlier number**. These were re-measured after that rebase, MiniMax-M3-Q4_K_M, CPU-only via a
wrapper that fails the run if any CUDA buffer appears (`-ngl 0` alone does not stop batch-GEMM
offload), `npp 16384`, `-ub 512`, `-t 48`, arms interleaved in one session on one binary:

| change | phase | effect | control |
|---|---|---:|---|
| `pool_1d` threading | prefill | **+27.95%** | dense control (`-npp 16384 -ntg 32`, the op is absent from that graph) −0.58%; greedy output bit-identical (42,118 bytes, same sha256) |
| CPU-FA kvsplit | decode | **+20.85%** | null control (`-npp 2048 -c 4096`) at `-t 16`, where the guard provably cannot fire, came out equal (−0.76%) |
| both together (2x2 factorial) | — | prefill **+26.95%**, decode **+21.26%** vs neither | interaction −1.09pp / −1.38pp, i.e. inside the prefill noise |

The 2x2 headline is *both patches versus neither*. Within that experiment kvsplit's own simple
effect is +18.54% (pool_1d on) and +19.92% (pool_1d off).

**Every arm above came from an uncommitted bench-only env toggle** (`IK_POOL1D_SERIAL`,
`IK_KVSPLIT_OFF`) so that one binary provides both A/B arms. The toggle is deliberately NOT on this
branch, so a checkout gives you the patched arm only, not an A/B. The two controls also came from
their own runs on other branches at build 4907, not from the 2x2's build 4908; their regimes are
stated with each control above.

### Earlier tables (PRE-REBASE — do not compare against the above)

The context-scaling, compute-buffer and quality tables previously in this document were measured on
build `4856 b2fa29c0`, **before** the rebase onto `7cff686d`. They are retained in git history.
Absolute throughput on the current base is roughly 10% higher on both arms, so those levels are
stale even where the ratios are not. They are not reproduced here to avoid being quoted as current.

## Quality

Sparse attention changes which cells are attended, so this needs a metric that survives an argmax
flip; greedy text does not.

**The CPU-FA kvsplit change is not output-preserving, and that is by construction.** It splits the
gathered rows across threads and combines partial `(M, S, R)` via `accumulate_qkv`, which
reassociates the softmax. Measured on the decode path at `-ub 1` (perplexity batches, so at the
default ubatch it exercises prefill and cannot see a decode-only change at all), against a private
held-out corpus:

| | floor (unpatched vs itself) | patched vs unpatched |
|---|---:|---:|
| same top-1 | **100.000 ± 0.000 %** | **98.437 ± 0.274 %** |
| maximum KLD | 0.000084 | 0.291754 |

The instrument is deterministic, so the 1.56% top-1 change is **real, not run-to-run noise**. Over
the same run perplexity did not move (ratio 0.99778 ± 0.00222, 1.000 inside 1σ) and mean Δp was
centred on zero — the signature of floating-point reassociation flipping near-ties rather than a
numerical defect. **This is one 4k chunk, n=1**; the error bars are within-chunk and understate the
true floor, so "no quality change detected" is the honest claim and "harmless" is not.

Earlier KLD figures for the mask-vs-gather comparison predate both the graph-reuse fix and the
rebase, and have not been re-measured.

## Known issues

- **The kvsplit change is not MSA-gated.** `iqk_fa_gather_kv_split` fires inside the generic
  `src[5]` index-list branch, which GLM-DSA (`build_deepseek2.cpp`) and DeepSeek-V4
  (`build_deepseek4.cpp`) also use. Its guard is `neq2 < nth && nkv/32 > 1 && nkv/32 >= neq2` with
  `neq2 = n_head / n_groups`, so on a machine with more threads than heads it will fire for those
  architectures too and apply the same softmax reassociation. If you run a DSA model with
  `-t > n_head`, this change is in your decode path.
- **The MSA branch of `build_k_shift` has never executed in any test here.** Every measurement used
  `llama-batched-bench` or `llama-perplexity`, neither of which shifts context, and no server run
  reached the context limit. The first long conversation that wraps will exercise it cold.
- `--msa` on an indexer-less GGUF logs as enabled and silently runs dense (above).
