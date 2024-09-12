IK_Llama.cpp.nxs is my fork of Ikawrakow's clone of Llama_cpp.

The objective of this fork is to adapt it to my quantization needs mainly, because Ikawrakow's quants are SOTA.
Then, to be able to use them on Croco.cpp, my fork of Kobold.cpp.
And also to amuse myself learning more about merging relatively complex stuff.

Current IK_Llama commit : 

First Llama.cpp mainline base : b3570 -> b3637 (pre GGML sync "File version V2")

Second Llama.cpp mainline base revision : b3638 -> b3643 (pre Threadpool take 2)

Third Llama.cpp mainline base : b3643 -> b3671 (pre LCPP Bitnet quantization TQ1_0 and TQ2_0)

Fourth Llama.cpp mainline base : b3671 -> b3680 (pre LCPP Refactor Sampling v2)

Fifth Llama.cpp mainline base : b3680 -> b3681 (LCPP Refactor Sampling v2)
- 2 IKL commits had to be reversed because I'm not able to port them.

Sixth Llama.cpp mainline base : b3681 -> b3683 (LCPP Common : refactor arg parser)
- IKL arguments had to be removed because I'm not able to port them.

Seventh Llama.cpp mainline base : b3683 -> b3715 (pre LCPP common : move arg parser code to `arg.cpp` (#9388))


----------


# llama.cpp clone with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a clone of [llama.cpp](https://github.com/ggerganov/llama.cpp) with the following improvements
* Better implementation of CPU matrix multiplications (`AVX2` and `ARM_NEON`) for `fp16/fp32` and all k-, i-, and legacy `llama.cpp` quants, that leads to a significant improvement in prompt processing (PP) speed, typically in the range of 2X, but up to 4X for some quantization types. Token generation (TG) also benefits, but to a lesser extent due to TG being memory bound
* Faster CPU inference for MoE models with similar performance gains
* Implementation of the [Bitnet b1.58](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) model for the CPU (`AVX2` and `ARM_NEON`) and GPU (`CUDA` and `Metal`). This implementation is much faster than the unmerged `llama.cpp` [PR-8151](https://github.com/ggerganov/llama.cpp/pull/8151)

If you are not already familiar with [llama.cpp](https://github.com/ggerganov/llama.cpp), it is better to start there. For those familiar with `llama.cpp`, everything here works the same as in `llama.cpp` (or at least the way `llama.cpp` worked when I last synced on Aug 12 2024).

Note that I have published some, but not all, of the code in this repository in a series of [llamafile](https://github.com/Mozilla-Ocho/llamafile) PRs ([394](https://github.com/Mozilla-Ocho/llamafile/pull/394), [405](https://github.com/Mozilla-Ocho/llamafile/pull/405), [428](https://github.com/Mozilla-Ocho/llamafile/pull/428), [435](https://github.com/Mozilla-Ocho/llamafile/pull/435), [453](https://github.com/Mozilla-Ocho/llamafile/pull/453), and [464](https://github.com/Mozilla-Ocho/llamafile/pull/464))

The implementation of matrix-matrix and matrix-vector multiplications is in a single C++ source file (`iqk_mul_mat.cpp`) with just two interface functions `iqk_mul_mat` (`fp16/fp32` and quantized matrix multiplications) and `iqk_mul_mat_moe` (as `iqk_mul_mat` but meant to be used for the FFN part of a MoE model). Under the hood `iqk_mul_mat_moe` uses the same implementation as `iqk_mul_mat`, with the only difference being where results are stored in memory. Bitnet quantization related stuff is in `iqk-quantize.cpp`.   

## Why?

Mostly out of curiosity:
* Justine Tunney's `tinyBLAS`, which she contributed to `llama.cpp` in [PR 6414](https://github.com/ggerganov/llama.cpp/pull/6414), only works for `Q4_0`, `Q8_0` and `fp16/bf16` models. In the surrounding discussion about possibly extending `tinyBLAS` to k- and i-quants, she felt that k-quants are [not amenable to block-tiling](https://github.com/ggerganov/llama.cpp/pull/6840#issuecomment-2072995387), which is required to improve performance. This statement piqued my curiosity, so here we are.
* Bitnet-1.58b has been one of the [most discussed topics](https://github.com/ggerganov/llama.cpp/issues/5761#issuecomment-2198380366) in the `llama.cpp` project, so eventually I decided to see how efficiently one can implement a ternary model

Curiosity aside, improved CPU performance may be (or may become) important in practice. According to The Register, 70% of AI inference [is done on the CPU of mobile phones](https://www.theregister.com/2024/05/30/arm_cortex_x925_ai_cores/?td=rt-3a), at least in the Android world (but I haven't come around to actually comparing performance on a phone). With ever increasing number of LLM model parameters, and with Meta's 400B model just released, the CPU may become the only viable option for people not willing (or not able to) rent/buy uber expensive GPU instances capable of running such models. Granted, one would need a pretty beefy computer to run a 400B model, and inference speed will be sluggish, but at least one will not need to spend the equivalent of a luxury apartment in the downtown of the city where I live to buy the GPU system capable of running the model.

## Performance comparison to llama.cpp

The results in the following tables are obtained with these parameters:
* Model is LLaMA-v3-8B for `AVX2` and LLaMA-v2-7B for `ARM_NEON`
* The `AVX2` CPU is a 16-core Ryzen-7950X
* The `ARM_NEON` CPU is M2-Max
* `tinyBLAS` is enabled in `llama.cpp`
* `llama.cpp` results are for `build: 081fe431 (3441)`, which was the current `llama.cpp` master branch when I pulled on July 23 2024.
* The projects are built without `CUDA` support, no `BLAS`, and Accelerate framework disabled

### Prompt processing

Here I set the number of threads to be equal to the number of (performance) cores of the CPU, so 16 threads for the Ryzen-7950X and 8 threads for the M2-Max. The following table summarizes the results. To not make the table too long, I have listed only quantized models containing predominantly one quantization type (i.e., excluded the `QX_K - Medium/Large` variants, which are typically a mix of `QX_K` and `Q(X+1)_K`, as well as `IQ2_S` and `IQ3_XS`).  

The command line to generate the benchmark data is
```
./bin/llama-bench -m $model -p 512 -n 0 -t $num_threads -ngl 0
```

| Quantization|       size | backend    | threads | t/s (llama.cpp)  | t/s (iqk_mul_mat)| Speedup |
| ----------- | ---------: | ---------- | ------: | ---------------: | ---------------: | ------: |
| 8B F16      |  14.96 GiB | AVX2       |      16 |    112.37 ± 0.40 |    131.27 ± 0.38 |  1.168  |
| 7B F16      |  12.55 GiB | NEON       |       8 |     90.28 ± 1.25 |     95.34 ± 0.15 |  1.056  |
| 8B Q8_0     |   7.95 GiB | AVX2       |      16 |    118.07 ± 0.53 |    134.00 ± 0.47 |  1.135  |
| 7B Q8_0     |   6.67 GiB | NEON       |       8 |     77.25 ± 1.81 |     94.14 ± 1.15 |  1.219  |
| 8B Q4_0     |   4.35 GiB | AVX2       |      16 |    104.46 ± 0.33 |    130.20 ± 0.29 |  1.246  |
| 7B Q4_0     |   3.57 GiB | NEON       |       8 |     65.46 ± 0.79 |     76.22 ± 0.71 |  1.164  |
| 8B Q4_1     |   4.77 GiB | AVX2       |      16 |     57.83 ± 0.24 |    160.69 ± 0.49 |  2.779  |
| 7B Q4_1     |   3.95 GiB | NEON       |       8 |     37.40 ± 0.50 |     65.83 ± 0.98 |  1.760  |
| 8B Q5_0     |   5.22 GiB | AVX2       |      16 |     53.50 ± 0.35 |    122.62 ± 0.48 |  2.292  |
| 7B Q5_0     |   4.34 GiB | NEON       |       8 |     29.31 ± 0.51 |     67.51 ± 1.17 |  2.303  |
| 8B Q5_1     |   5.64 GiB | AVX2       |      16 |     50.85 ± 0.36 |    147.15 ± 0.47 |  2.894  |
| 7B Q5_1     |   4.72 GiB | NEON       |       8 |     26.02 ± 0.37 |     58.49 ± 0.85 |  2.248  |
| 8B Q2_K_S   |   2.78 GiB | AVX2       |      16 |    110.11 ± 0.28 |    192.47 ± 1.35 |  1.748  |
| 7B Q2_K_S   |   2.16 GiB | NEON       |       8 |     35.44 ± 0.06 |     77.93 ± 1.64 |  2.199  |
| 8B Q3_K_S   |   3.41 GiB | AVX2       |      16 |     77.42 ± 0.36 |    181.64 ± 0.44 |  2.346  |
| 7B Q3_K_S   |   2.75 GiB | NEON       |       8 |     26.79 ± 0.03 |     59.38 ± 1.08 |  2.216  |
| 8B Q4_K_S   |   4.36 GiB | AVX2       |      16 |     98.92 ± 0.34 |    185.35 ± 0.39 |  1.874  |
| 7B Q4_K_S   |   3.59 GiB | NEON       |       8 |     46.55 ± 0.67 |     76.31 ± 0.38 |  1.639  |
| 8B Q5_K_S   |   5.21 GiB | AVX2       |      16 |     69.44 ± 0.31 |    179.62 ± 0.69 |  2.587  |
| 7B Q5_K_S   |   4.33 GiB | NEON       |       8 |     30.18 ± 0.23 |     65.34 ± 0.79 |  2.165  |
| 8B Q6_K     |   6.14 GiB | AVX2       |      16 |     74.89 ± 0.26 |    181.86 ± 0.55 |  2.428  |
| 7B Q6_K     |   5.15 GiB | NEON       |       8 |     28.12 ± 1.24 |     60.75 ± 1.15 |  2.160  |
| 8B IQ2_XXS  |   2.23 GiB | AVX2       |      16 |     42.57 ± 0.16 |    126.63 ± 0.55 |  2.975  |
| 7B IQ2_XXS  |   1.73 GiB | NEON       |       8 |     20.87 ± 0.20 |     64.29 ± 1.12 |  3.080  |
| 8B IQ2_XS   |   2.42 GiB | AVX2       |      16 |     46.45 ± 0.27 |    125.46 ± 0.43 |  2.701  |
| 7B IQ2_XS   |   1.89 GiB | NEON       |       8 |     22.77 ± 0.21 |     51.15 ± 0.24 |  2.246  |
| 8B IQ2_M    |   2.74 GiB | AVX2       |      16 |     40.76 ± 0.18 |    113.07 ± 0.48 |  2.774  |
| 7B IQ2_M    |   2.20 GiB | NEON       |       8 |     14.95 ± 0.26 |     44.87 ± 0.50 |  3.001  |
| 8B IQ3_XXS  |   3.04 GiB | AVX2       |      16 |     31.95 ± 0.20 |    109.86 ± 0.45 |  3.438  |
| 7B IQ3_XXS  |   2.41 GiB | NEON       |       8 |     14.40 ± 0.10 |     53.58 ± 0.85 |  3.721  |
| 8B IQ3_S    |   3.42 GiB | AVX2       |      16 |     28.04 ± 0.08 |     96.28 ± 0.45 |  3.434  |
| 7B IQ3_S    |   2.75 GiB | NEON       |       8 |     12.08 ± 0.30 |     49.72 ± 0.06 |  4.116  |
| 8B IQ4_XS   |   4.13 GiB | AVX2       |      16 |     68.98 ± 0.31 |    180.34 ± 0.55 |  2.614  |
| 7B IQ4_XS   |   3.37 GiB | NEON       |       8 |     40.67 ± 1.97 |     75.11 ± 1.97 |  1.847  |
| 8B IQ4_NL   |   4.35 GiB | AVX2       |      16 |     59.94 ± 0.21 |    129.06 ± 0.43 |  2.153  |
| 7B IQ4_NL   |   3.56 GiB | NEON       |       8 |     34.36 ± 0.81 |     76.02 ± 1.36 |  2.212  |

We see that `llama.cpp` achieves respectable performance for `fp16`, `Q8_0`, and `Q4_0`, being only up to 25% slower than this implementation. This is thanks to the use of Justine Tunney's `tinyBLAS`, which is utilized for these quantization types. For all other quants we observe performance gains in the `1.75X - 4X` range, which is not a small feat considering that the `ggml` matrix multiplication functions has been rewritten several times since `llama.cpp` was first published. Performance gains are larger for i-quants due to the higher quant unpacking cost (see discussion in "To tile or not to tile")

### Token generation

On the Ryzen-7950X TG is memory bound, and for many quantization types peak performance is achieved at just 4 threads. Hence, only results for 2 and 4 threads are shown for `AVX2`. The M2-Max has a much more capable memory subsystem and as a result performance keep increasing up to 8 threads. Thus, results are given for up to 8 threads for `ARM_NEON`.

The command line to generate the data was
```
./bin/llama-bench -m $model -p 0 -n 128 -t $num_threads -ngl 0
```

| Quantization|       size | backend    | threads | t/s (llama.cpp)  | t/s (iqk_mul_mat)| Speedup |
| ---------- | ---------: | ---------- | ------: | ---------------: | ---------------: | ------: |
| 8B F16     |  14.96 GiB | AVX2       |       1 |      2.20 ± 0.00 |      2.25 ± 0.00 |  1.023  |
|            |            |            |       2 |      3.63 ± 0.00 |      3.68 ± 0.00 |  1.014  |
|            |            |            |       4 |      4.20 ± 0.00 |      4.20 ± 0.00 |  1.000  |
| 7B F16     |  12.55 GiB | NEON       |       2 |      6.94 ± 0.27 |      7.40 ± 0.01 |  1.066  |
|            |            |            |       4 |      8.73 ± 0.01 |      8.83 ± 0.01 |  1.011  |
|            |            |            |       6 |      9.05 ± 0.02 |      9.05 ± 0.01 |  1.000  |
| 8B Q8_0    |   7.95 GiB | AVX2       |       2 |      5.03 ± 0.00 |      7.87 ± 0.00 |  1.565  |
|            |            |            |       4 |      7.40 ± 0.00 |      7.82 ± 0.00 |  1.057  |
| 7B Q8_0    |   6.67 GiB | NEON       |       2 |      8.29 ± 0.44 |     12.07 ± 0.10 |  1.456  |
|            |            |            |       4 |     13.53 ± 0.03 |     15.77 ± 0.08 |  1.166  |
|            |            |            |       8 |     16.24 ± 0.10 |     16.94 ± 0.04 |  1.043  |
| 8B Q4_0    |   4.35 GiB | AVX2       |       2 |      6.36 ± 0.00 |     10.28 ± 0.00 |  1.616  |
|            |            |            |       4 |     10.97 ± 0.06 |     13.55 ± 0.07 |  1.235  |
| 7B Q4_0    |   3.57 GiB | NEON       |       2 |      9.77 ± 0.02 |     13.69 ± 0.03 |  1.401  |
|            |            |            |       4 |     17.82 ± 0.06 |     23.98 ± 0.11 |  1.346  |
|            |            |            |       8 |     26.63 ± 0.41 |     29.86 ± 0.04 |  1.121  |
| 8B Q4_1    |   4.77 GiB | AVX2       |       2 |      5.11 ± 0.00 |     11.45 ± 0.00 |  2.241  |
|            |            |            |       4 |      9.08 ± 0.02 |     12.58 ± 0.00 |  1.385  |
| 7B Q4_1    |   3.95 GiB | NEON       |       2 |      9.11 ± 0.06 |     14.62 ± 0.04 |  1.605  |
|            |            |            |       4 |     17.04 ± 0.09 |     24.08 ± 0.28 |  1.413  |
|            |            |            |       8 |     25.26 ± 0.24 |     27.23 ± 0.14 |  1.078  |
| 8B Q5_0    |   5.22 GiB | AVX2       |       2 |      5.31 ± 0.01 |      8.30 ± 0.01 |  1.563  |
|            |            |            |       4 |      9.40 ± 0.01 |     11.47 ± 0.00 |  1.220  |
| 7B Q5_0    |   4.34 GiB | NEON       |       2 |      7.26 ± 0.06 |      7.52 ± 0.00 |  1.036  |
|            |            |            |       4 |     13.63 ± 0.18 |     14.16 ± 0.10 |  1.039  |
|            |            |            |       8 |     22.55 ± 0.35 |     24.34 ± 0.22 |  1.079  |
| 8B Q5_1    |   5.64 GiB | AVX2       |       2 |      4.52 ± 0.00 |      8.86 ± 0.00 |  1.960  |
|            |            |            |       4 |      7.72 ± 0.05 |     10.68 ± 0.03 |  1.383  |
| 7B Q5_1    |   4.72 GiB | NEON       |       2 |      6.51 ± 0.01 |      6.42 ± 0.03 |  0.986  |
|            |            |            |       4 |     12.26 ± 0.18 |     12.21 ± 0.14 |  0.996  |
|            |            |            |       8 |     20.33 ± 0.52 |     21.85 ± 0.22 |  1.075  |
| 8B Q2_K_S  |   2.78 GiB | AVX2       |       2 |     11.30 ± 0.00 |     13.06 ± 0.01 |  1.156  |
|            |            |            |       4 |     18.70 ± 0.00 |     19.04 ± 0.65 |  1.014  |
| 7B Q2_K_S  |   2.16 GiB | NEON       |       2 |      8.42 ± 0.05 |     11.97 ± 0.10 |  1.422  |
|            |            |            |       4 |     15.74 ± 0.01 |     22.09 ± 0.08 |  1.403  |
|            |            |            |       8 |     27.35 ± 0.05 |     38.32 ± 0.05 |  1.401  |
| 8B Q3_K_S  |   3.41 GiB | AVX2       |       2 |      8.58 ± 0.00 |     10.82 ± 0.00 |  1.261  |
|            |            |            |       4 |     15.26 ± 0.01 |     16.25 ± 0.01 |  1.065  |
| 7B Q3_K_S  |   2.75 GiB | NEON       |       2 |      6.40 ± 0.02 |      9.12 ± 0.09 |  1.425  |
|            |            |            |       4 |     12.17 ± 0.00 |     17.11 ± 0.03 |  1.406  |
|            |            |            |       8 |     22.04 ± 0.08 |     31.39 ± 0.31 |  1.424  |
| 8B Q4_K_S  |   4.36 GiB | AVX2       |       2 |      9.61 ± 0.00 |     10.72 ± 0.01 |  1.116  |
|            |            |            |       4 |     13.24 ± 0.31 |     13.28 ± 0.01 |  1.003  |
| 7B Q4_K_S  |   3.59 GiB | NEON       |       2 |     11.15 ± 0.05 |     12.93 ± 0.09 |  1.160  |
|            |            |            |       4 |     20.24 ± 0.16 |     23.49 ± 0.29 |  1.161  |
|            |            |            |       8 |     25.76 ± 0.07 |     28.31 ± 0.22 |  1.099  |
| 8B Q5_K_S  |   5.21 GiB | AVX2       |       2 |      7.45 ± 0.00 |      9.73 ± 0.00 |  1.306  |
|            |            |            |       4 |     11.05 ± 0.33 |     11.43 ± 0.02 |  1.034  |
| 7B Q5_K_S  |   4.33 GiB | NEON       |       2 |      7.20 ± 0.04 |      8.81 ± 0.04 |  1.224  |
|            |            |            |       4 |     13.62 ± 0.15 |     16.81 ± 0.16 |  1.234  |
|            |            |            |       8 |     20.56 ± 0.19 |     23.96 ± 0.14 |  1.165  |
| 8B Q6_K    |   6.14 GiB | AVX2       |       2 |      7.53 ± 0.00 |      9.42 ± 0.00 |  1.251  |
|            |            |            |       4 |      9.74 ± 0.00 |      9.97 ± 0.01 |  1.024  |
| 7B Q6_K    |   5.15 GiB | NEON       |       2 |      6.85 ± 0.04 |      8.30 ± 0.06 |  1.212  |
|            |            |            |       4 |     13.03 ± 0.05 |     15.47 ± 0.17 |  1.187  |
|            |            |            |       8 |     18.52 ± 0.07 |     20.67 ± 0.08 |  1.116  |
| 8B IQ2_XXS |   2.23 GiB | AVX2       |       2 |      5.33 ± 0.01 |      6.40 ± 0.00 |  1.201  |
|            |            |            |       4 |     10.06 ± 0.03 |     11.76 ± 0.03 |  1.169  |
| 7B IQ2_XXS |   1.73 GiB | NEON       |       2 |      5.07 ± 0.04 |      5.22 ± 0.05 |  1.030  |
|            |            |            |       4 |      9.63 ± 0.00 |      9.91 ± 0.07 |  1.029  |
|            |            |            |       8 |     17.40 ± 0.50 |     18.65 ± 0.22 |  1.072  |
| 8B IQ2_XS  |   2.42 GiB | AVX2       |       2 |      5.83 ± 0.00 |      6.55 ± 0.00 |  1.123  |
|            |            |            |       4 |     10.88 ± 0.09 |     12.07 ± 0.07 |  1.109  |
| 7B IQ2_XS  |   1.89 GiB | NEON       |       2 |      5.52 ± 0.01 |      5.60 ± 0.00 |  1.014  |
|            |            |            |       4 |     10.50 ± 0.01 |     11.15 ± 0.00 |  1.062  |
|            |            |            |       8 |     18.19 ± 1.30 |     20.94 ± 0.19 |  1.151  |
| 8B IQ2_M   |   2.74 GiB | AVX2       |       2 |      5.12 ± 0.01 |      5.17 ± 0.00 |  1.010  |
|            |            |            |       4 |      9.60 ± 0.28 |      9.68 ± 0.16 |  1.008  |
| 7B IQ2_M   |   2.20 GiB | NEON       |       2 |      3.73 ± 0.02 |      4.53 ± 0.00 |  1.214  |
|            |            |            |       4 |      7.14 ± 0.05 |      8.70 ± 0.06 |  1.218  |
|            |            |            |       8 |     11.99 ± 0.48 |     16.41 ± 0.05 |  1.369  |
| 8B IQ3_XXS |   3.04 GiB | AVX2       |       2 |      4.06 ± 0.01 |      5.00 ± 0.00 |  1.232  |
|            |            |            |       4 |      7.75 ± 0.02 |      9.13 ± 0.45 |  1.178  |
| 7B IQ3_XXS |   2.41 GiB | NEON       |       2 |      3.53 ± 0.00 |      3.82 ± 0.00 |  1.082  |
|            |            |            |       4 |      6.74 ± 0.04 |      7.42 ± 0.07 |  1.103  |
|            |            |            |       8 |     11.96 ± 0.40 |     13.19 ± 0.29 |  1.103  |
| 8B IQ3_S   |   3.42 GiB | AVX2       |       2 |      3.62 ± 0.00 |      4.06 ± 0.00 |  1.122  |
|            |            |            |       4 |      6.80 ± 0.01 |      7.62 ± 0.10 |  1.121  |
| 7B IQ3_S   |   2.75 GiB | NEON       |       2 |      2.96 ± 0.01 |      3.21 ± 0.03 |  1.084  |
|            |            |            |       4 |      5.68 ± 0.01 |      6.25 ± 0.05 |  1.100  |
|            |            |            |       8 |     10.32 ± 0.25 |     11.11 ± 0.37 |  1.077  |
| 8B IQ4_XS  |   4.13 GiB | AVX2       |       2 |      8.08 ± 0.00 |     11.35 ± 0.00 |  1.405  |
|            |            |            |       4 |     13.36 ± 0.72 |     14.32 ± 0.24 |  1.072  |
| 7B IQ4_XS  |   3.37 GiB | NEON       |       2 |      9.87 ± 0.03 |     12.06 ± 0.00 |  1.222  |
|            |            |            |       4 |     17.78 ± 0.23 |     22.06 ± 0.28 |  1.241  |
|            |            |            |       8 |     27.62 ± 0.09 |     29.70 ± 0.39 |  1.075  |
| 8B IQ4_NL  |   4.35 GiB | AVX2       |       2 |      5.52 ± 0.00 |     10.26 ± 0.00 |  1.859  |
|            |            |            |       4 |     10.78 ± 0.01 |     13.69 ± 0.08 |  1.270  |
| 7B IQ4_NL  |   3.56 GiB | NEON       |       2 |      8.32 ± 0.01 |     13.54 ± 0.01 |  1.627  |
|            |            |            |       4 |     15.89 ± 0.00 |     24.28 ± 0.29 |  1.528  |
|            |            |            |       8 |     26.56 ± 0.36 |     29.87 ± 0.08 |  1.125  |

Here gains are generally lower compared to PP due to TG performance being limited by memory bandwidth. Nevertheless, for some quants/architectures/threads the speedup is quite remarkable (e.g., almost a factor of 2 for `Q5_1` on `AVX2` with 2 threads).  

## MoE models

There is [PR-6840](https://github.com/ggerganov/llama.cpp/pull/6840) from Justine Tunney in `llama.cpp`, but it has not been merged since April 23, so I'll compare performance to the master branch for Mixtral-8x7B. As Mixtral8x7B quantization is quite a lengthy process, the following table shows data only for `Q4_K_S` (a commonly used k-quant, 4 bit), `Q5_0` (a legacy quant, 5 bit), and `IQ4_XXS` (a 3-bit i-quant)

| model        |       size | backend    | threads |     test |  t/s (llama.cpp) | t/s (iqk_mul_mat)| Speedup |
| ------------ | ---------: | ---------- | ------: | -------: | ---------------: | ---------------: | ------: |
| 8x7B Q4_K_S  |  48.75 GiB | AVX2       |      16 |    pp512 |     54.92 ± 0.23 |    102.94 ± 0.37 |  1.874  |
|              |            | NEON       |       8 |    pp512 |     23.54 ± 1.56 |     38.32 ± 0.54 |  1.628  |
|              |            | AVX2       |       4 |    tg128 |      7.80 ± 0.07 |      7.83 ± 0.09 |  1.004  |
|              |            | NEON       |       8 |    tg128 |     14.95 ± 0.25 |     15.28 ± 0.24 |  2.022  |
| 8x7B IQ3_XXS |  33.07 GiB | AVX2       |      16 |    pp512 |     17.58 ± 0.04 |     68.45 ± 0.22 |  3.894  |
|              |            | NEON       |       8 |    pp512 |      7.75 ± 0.04 |     34.67 ± 0.40 |  4.474  |
|              |            | AVX2       |       4 |    tg128 |      4.60 ± 0.01 |      5.45 ± 0.09 |  1.185  |
|              |            | AVX2       |       8 |    tg128 |      8.04 ± 0.65 |      9.83 ± 0.06 |  1.223  |
|              |            | AVX2       |      16 |    tg128 |     10.42 ± 0.01 |     10.57 ± 0.01 |  1.014  |
|              |            | NEON       |       8 |    tg128 |      6.19 ± 1.16 |      7.27 ± 0.14 |  1.174  |
| 8x7B Q5_0    |  59.11 GiB | AVX2       |      16 |    pp512 |     29.06 ± 0.43 |     62.67 ± 0.32 |  2.157  |
|              |            | NEON       |       8 |    pp512 |     15.17 ± 0.51 |     27.36 ± 1.03 |  1.804  |
|              |            | AVX2       |       4 |    tg128 |      5.44 ± 0.10 |      6.81 ± 0.06 |  1.252  |
|              |            | NEON       |       8 |    tg128 |     12.03 ± 0.77 |     12.41 ± 1.27 |  1.032  |


## Bitnet-1.58B

Two implementations are provided
* `IQ1_BN` - uses 1.625 bits-per-weight (bpw)
* `IQ2_BN` - uses 2.0 bpw

`IQ2_BN` is faster for PP (CPU and GPU, although the PP performance difference on CUDA is very minor). `IQ1_BN` can arrive at a higher TG performance on the Ryzen-7950X (given enough threads) because of the smaller model size, but it is always slower on the GPU and on the M2-Max CPU.

There is the unmerged [PR 8151](https://github.com/ggerganov/llama.cpp/pull/8151) in `llama.cpp` that implements Bitnet-1.58B for the CPU (`AVX` and `ARM_NEON`, no GPU implementation). The following table compares performance between this repo and `PR-8151` in `llama.cpp`. The CUDA results were obtained on an RTX-4080, the Metal results on a 30-core M2-Max GPU.

| model       |       size | backend    | threads |   test | t/s (llama.cpp)  | t/s (this repo)| Speedup |
| ----------- | ---------: | ---------- | ------: | -----: | ---------------: | -------------: | ------: |
| 3B - IQ1_BN | 729.64 MiB | AVX2       |      16 |  pp512 |    120.61 ± 0.48 | 423.19 ± 1.28  |  3.509  |
|             |            | NEON       |       8 |  pp512 |     46.64 ± 0.02 | 205.90 ± 0.88  |  4.415  |
|             |            | CUDA       |       8 |  pp512 |           -      | 10660 ± 170    |    -    |
|             |            | Metal      |       8 |  pp512 |           -      | 698.25 ± 1.91  |    -    |
|             |            | AVX2       |       2 |  tg128 |     15.79 ± 0.01 |  22.13 ± 0.02  |  1.402  |
|             |            | AVX2       |       4 |  tg128 |     28.64 ± 1.72 |  40.14 ± 0.04  |  1.402  |
|             |            | AVX2       |       8 |  tg128 |     48.91 ± 0.08 |  61.79 ± 0.09  |  1.263  |
|             |            | AVX2       |      16 |  tg128 |     57.73 ± 0.05 |  60.79 ± 0.05  |  1.053  |
|             |            | NEON       |       2 |  tg128 |     11.43 ± 0.04 |  16.87 ± 0.02  |  1.476  |
|             |            | NEON       |       4 |  tg128 |     21.11 ± 0.05 |  30.66 ± 0.11  |  1.452  |
|             |            | NEON       |       8 |  tg128 |     37.36 ± 0.07 |  55.21 ± 0.16  |  1.478  |
|             |            | CUDA       |       8 |  tg128 |           -      | 301.44 ± 0.12  |    -    |
|             |            | Metal      |       8 |  tg128 |           -      |  76.70 ± 0.07  |    -    |
| 3B - IQ2_BN | 873.65 MiB | AVX2       |      16 |  pp512 |    151.39 ± 0.35 | 540.82 ± 2.48  |  3.572  |
|             |            | NEON       |       8 |  pp512 |     46.54 ± 0.03 | 242.05 ± 0.34  |  5.201  |
|             |            | CUDA       |       8 |  pp512 |           -      | 10800 ± 160    |    -    |
|             |            | Metal      |       8 |  pp512 |           -      | 723.19 ± 0.53  |    -    |
|             |            | AVX2       |       2 |  tg128 |     18.93 ± 0.02 |  38.34 ± 0.08  |  2.026  |
|             |            | AVX2       |       4 |  tg128 |     34.54 ± 0.06 |  56.29 ± 0.07  |  1.630  |
|             |            | AVX2       |       8 |  tg128 |     52.97 ± 0.07 |  53.44 ± 0.08  |  1.009  |
|             |            | AVX2       |      16 |  tg128 |     51.84 ± 0.25 |  53.46 ± 0.07  |  1.031  |
|             |            | NEON       |       2 |  tg128 |     11.40 ± 0.02 |  32.01 ± 0.27  |  2.808  |
|             |            | NEON       |       4 |  tg128 |     20.99 ± 0.00 |  56.45 ± 0.11  |  2.689  |
|             |            | NEON       |       8 |  tg128 |     37.28 ± 0.08 |  89.77 ± 0.70  |  2.408  |
|             |            | CUDA       |       8 |  tg128 |           -      | 322.10 ± 0.07  |    -    |
|             |            | Metal      |       8 |  tg128 |           -      | 110.39 ± 0.13  |    -    |

We can make the following observations:
* For prompt processing this Bitnet-1.58b implementation is massively better than PR-8151 in `llama.cpp`, with gains between 3.4X and 5.2X!
* We get `PP-512 = 520 t/s` for the 2.0 bpw variant on the Ryzen-7950X, which costs less than $500. Hey, who needs a GPU?  
* For low number of threads (2), this implementation is also much faster than PR-8151 for TG, where speed gains are between 1.4X and 2.8X. As we become memory bound on the Ryzen-7950X, the speed advantage goes away there for sufficiently high number of threads. But on the M2-Max this implementation is 1.4X (1.625 bpw) or 2.4X faster even at 8 threads
* Looking at TG on the M2-Max, the GPU looks a bit like wasted silicon (90 vs 110 t/s for TG-128 and the 2.0 bpw variant). If the GPU transistors had been spent to double the M2 number of CPU cores (and all memory bandwidth is given to the CPU), the CPU would be wiping the floor with the GPU.
* I'm of course kidding with the above. Still, it seems there are massive inefficiencies in the `llama.cpp` Metal implementation that start showing up when matrix multiplications become very fast as is the case here. The difference between CPU and GPU prompt processing speed is typically at least a factor of 7 in favor of the GPU on the M2-Max, but it is only around a factor of 3 here.
* It is worth noting that one needs to offload the token embeddings tensor to the GPU, else performance on CUDA/Metal is significantly lower. Bitnet uses the same tensor for token embeddings and for output. Mainline `llama.cpp` currently puts the token embeddings tensor on the CPU, and this results in running the matrix multiplication with the output tensor on the CPU. This most likely affects other models as well (e.g., Gemma), but I haven't yet looked into this.

To reproduce these results:
* Clone https://huggingface.co/1bitLLM/bitnet_b1_58-3B
* Run `python3 --outtype f16 path_to_bitnet` to convert to GGUF
* Run `./bin/llama-quantize path_to_bitnet/ggml-model-f16.gguf quantized.gguf [iq1_bn | iq2_bn]`. Note: no imatrix is required (and, if you provide one, it is ignored)
* Caveat: only the 3B Bitnet variant works. The smaller Bitnet models contain tensors with number of columns that are not even a multiple of 32, so basically no `llama.cpp` quant will work for these.  

## To tile or not to tile

The common wisdom for efficient matrix multiplications is to use block tiling, and this is also used here for `fp16/fp32` matrices. But block tiling does not somehow magically reduce the amount of computation that needs to get done. Performance gains are simply due to the better utilization of memory caches. When dealing with quantized matrix multiplications, there is an additional factor that comes into play: the quantized data needs to be unpacked to 8-bit integers before being used in the matrix multiplication multiply-add operations. Depending on quantization type, this unpacking can represent a significant fraction of the overall computation cost. Hence, for best performance, one would want to reuse the unpacked quants as much as possible, thus spending some fraction of the available vector registers to hold the unpacked data. But when using block tiling, one also needs a certain number of vector registers for accumulating results. For instance, on `AVX2` (16 vector registers available), for `fp16/fp32` models best performance is achieved with `2 x 6` tiles (where the `2` refers to rows in the left matrix and is measured in units of the vector register size, so 16/8 floats for `fp16/fp32`, and `6` is for the number of columns in the right matrix). Unpacking quantized data works best when done in blocks of 128 or 256 quants so that, if we wanted to keep unpacked quants for 2 rows, we would need at least 8 vector registers, thus being left with less than 8 registers for result accumulation, so at best `2 x 3` tiles. In practice one needs addition vector registers for various constants that are typically needed for de-quantization, so that, at the end, it becomes better to use `1 x N` "tiles", i.e., a row-wise multiplication where each row in the left matrix is multiplied with `N` columns in the right matrix, thus reusing the unpacked data `N` times. This (i.e., amortizing de-quantization cost) is the main mechanism for seeding up quantized matrix multiplications. Having started with quantized matrices, and having gone from tiles to a row-wise implementation after some experimentation, I did try row-wise multiplication for float matrices first. Performance was not quite as good as for block-tiling, but I did get up to 90-95% of the speed of `tinyBLAS` that way before switching the `fp16/fp32` implementation to `2 x 6` (`AVX2`) or `5 x 5` (`AVX512` and `ARM_NEON`) block-tiles. But even for for `Q8_0 x Q8_0` multiplications, where there is basically no de-quantization cost, row-wise multiplication is faster than tiling (and hence this implemeintation beats `tinyBLAS`, which uses block-tiling also for `Q8_0`).



----------



# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Server](https://github.com/ggerganov/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggerganov/llama.cpp/actions/workflows/server.yml)
[![Conan Center](https://shields.io/conan/v/llama-cpp)](https://conan.io/center/llama-cpp)

[Roadmap](https://github.com/users/ggerganov/projects/7) / [Project status](https://github.com/ggerganov/llama.cpp/discussions/3471) / [Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205) / [ggml](https://github.com/ggerganov/ggml)

Inference of Meta's [LLaMA](https://arxiv.org/abs/2302.13971) model (and others) in pure C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggerganov/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggerganov/llama.cpp/issues/9291)

## Hot topics

- Huggingface GGUF editor: [discussion](https://github.com/ggerganov/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
variety of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2 and AVX512 support for x86 architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

Since its [inception](https://github.com/ggerganov/llama.cpp/issues/33#issuecomment-1465108022), the project has
improved significantly thanks to many contributions. It is the main playground for developing new features for the
[ggml](https://github.com/ggerganov/ggml) library.

**Supported models:**

Typically finetunes of the base models below are supported as well.

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggerganov/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggerganov/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggerganov/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggerganov/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggerganov/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggerganov/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)

(instructions for supporting more models: [HOWTO-add-model.md](./docs/development/HOWTO-add-model.md))

**Multimodal models:**

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)

**Bindings:**

- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggerganov/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)

**UI:**

Unless otherwise noted these projects are open-source with permissive licensing:

- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [nat/openplayground](https://github.com/nat/openplayground)
- [Faraday](https://faraday.dev/) (proprietary)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [Layla](https://play.google.com/store/apps/details?id=com.laylalite) (proprietary)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)
- [ollama/ollama](https://github.com/ollama/ollama)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [RAGNA Desktop](https://ragna.app/) (proprietary)
- [RecurseChat](https://recurse.chat/) (proprietary)
- [semperai/amica](https://github.com/semperai/amica)
- [withcatai/catai](https://github.com/withcatai/catai)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Msty](https://msty.app) (proprietary)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file)(Apachev2.0 or later)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [MindMac](https://mindmac.app) (proprietary)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [AIKit](https://github.com/sozercan/aikit) (MIT)
- [LARS - The LLM & Advanced Referencing Solution](https://github.com/abgulati/LARS) (AGPL)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

**Tools:**

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage

**Infrastructure:**

- [Paddler](https://github.com/distantmagic/paddler) - Stateful load balancer custom-tailored for llama.cpp
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs

**Games:**
- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

## Demo

<details>
<summary>Typical run using LLaMA v2 13B on M2 Ultra</summary>

```
$ make -j && ./llama-cli -m models/llama-13b-v2/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e
I llama.cpp build info:
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.            -O3 -std=c11   -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -pthread -DGGML_USE_K_QUANTS -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./common -O3 -std=c++11 -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -pthread -DGGML_USE_K_QUANTS
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.3 (clang-1403.0.22.14.1)
I CXX:      Apple clang version 14.0.3 (clang-1403.0.22.14.1)

make: Nothing to be done for `default'.
main: build = 1041 (cf658ad)
main: seed  = 1692823051
llama_model_loader: loaded meta data with 16 key-value pairs and 363 tensors from models/llama-13b-v2/ggml-model-q4_0.gguf (version GGUF V1 (latest))
llama_model_loader: - type  f32:   81 tensors
llama_model_loader: - type q4_0:  281 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_print_meta: format         = GGUF V1 (latest)
llm_load_print_meta: arch           = llama
llm_load_print_meta: vocab type     = SPM
llm_load_print_meta: n_vocab        = 32000
llm_load_print_meta: n_merges       = 0
llm_load_print_meta: n_ctx_train    = 4096
llm_load_print_meta: n_ctx          = 512
llm_load_print_meta: n_embd         = 5120
llm_load_print_meta: n_head         = 40
llm_load_print_meta: n_head_kv      = 40
llm_load_print_meta: n_layer        = 40
llm_load_print_meta: n_rot          = 128
llm_load_print_meta: n_gqa          = 1
llm_load_print_meta: f_norm_eps     = 1.0e-05
llm_load_print_meta: f_norm_rms_eps = 1.0e-05
llm_load_print_meta: n_ff           = 13824
llm_load_print_meta: freq_base      = 10000.0
llm_load_print_meta: freq_scale     = 1
llm_load_print_meta: model type     = 13B
llm_load_print_meta: model ftype    = mostly Q4_0
llm_load_print_meta: model size     = 13.02 B
llm_load_print_meta: general.name   = LLaMA v2
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MB
llm_load_tensors: mem required  = 7024.01 MB (+  400.00 MB per state)
...................................................................................................
llama_new_context_with_model: kv self size  =  400.00 MB
llama_new_context_with_model: compute buffer total size =   75.41 MB

system_info: n_threads = 16 / 24 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | VSX = 0 |
sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, n_batch = 512, n_predict = 400, n_keep = 0


 Building a website can be done in 10 simple steps:
Step 1: Find the right website platform.
Step 2: Choose your domain name and hosting plan.
Step 3: Design your website layout.
Step 4: Write your website content and add images.
Step 5: Install security features to protect your site from hackers or spammers
Step 6: Test your website on multiple browsers, mobile devices, operating systems etc…
Step 7: Test it again with people who are not related to you personally – friends or family members will work just fine!
Step 8: Start marketing and promoting the website via social media channels or paid ads
Step 9: Analyze how many visitors have come to your site so far, what type of people visit more often than others (e.g., men vs women) etc…
Step 10: Continue to improve upon all aspects mentioned above by following trends in web design and staying up-to-date on new technologies that can enhance user experience even further!
How does a Website Work?
A website works by having pages, which are made of HTML code. This code tells your computer how to display the content on each page you visit – whether it’s an image or text file (like PDFs). In order for someone else’s browser not only be able but also want those same results when accessing any given URL; some additional steps need taken by way of programming scripts that will add functionality such as making links clickable!
The most common type is called static HTML pages because they remain unchanged over time unless modified manually (either through editing files directly or using an interface such as WordPress). They are usually served up via HTTP protocols – this means anyone can access them without having any special privileges like being part of a group who is allowed into restricted areas online; however, there may still exist some limitations depending upon where one lives geographically speaking.
How to
llama_print_timings:        load time =   576.45 ms
llama_print_timings:      sample time =   283.10 ms /   400 runs   (    0.71 ms per token,  1412.91 tokens per second)
llama_print_timings: prompt eval time =   599.83 ms /    19 tokens (   31.57 ms per token,    31.68 tokens per second)
llama_print_timings:        eval time = 24513.59 ms /   399 runs   (   61.44 ms per token,    16.28 tokens per second)
llama_print_timings:       total time = 25431.49 ms
```

</details>

<details>
<summary>Demo of running both LLaMA-7B and whisper.cpp on a single M1 Pro MacBook</summary>

And here is another demo of running both LLaMA-7B and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) on a single M1 Pro MacBook:

https://user-images.githubusercontent.com/1991296/224442907-7693d4be-acaa-4e01-8b4f-add84093ffff.mp4

</details>

## Usage

Here are the end-to-end binary build and model conversion steps for most supported models.

### Basic usage

Firstly, you need to get the binary. There are different methods that you can follow:
- Method 1: Clone this repository and build locally, see [how to build](./docs/build.md)
- Method 2: If you are using MacOS or Linux, you can install llama.cpp via [brew, flox or nix](./docs/install.md)
- Method 3: Use a Docker image, see [documentation for Docker](./docs/docker.md)
- Method 4: Download pre-built binary from [releases](https://github.com/ggerganov/llama.cpp/releases)

You can run a basic completion using this command:

```bash
llama-cli -m your_model.gguf -p "I believe the meaning of life is" -n 128

# Output:
# I believe the meaning of life is to find your own truth and to live in accordance with it. For me, this means being true to myself and following my passions, even if they don't align with societal expectations. I think that's what I love about yoga – it's not just a physical practice, but a spiritual one too. It's about connecting with yourself, listening to your inner voice, and honoring your own unique journey.
```

See [this page](./examples/main/README.md) for a full list of parameters.

### Conversation mode

If you want a more ChatGPT-like experience, you can run in conversation mode by passing `-cnv` as a parameter:

```bash
llama-cli -m your_model.gguf -p "You are a helpful assistant" -cnv

# Output:
# > hi, who are you?
# Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
#
# > what is 1+1?
# Easy peasy! The answer to 1+1 is... 2!
```

By default, the chat template will be taken from the input model. If you want to use another chat template, pass `--chat-template NAME` as a parameter. See the list of [supported templates](https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)

```bash
./llama-cli -m your_model.gguf -p "You are a helpful assistant" -cnv --chat-template chatml
```

You can also use your own template via in-prefix, in-suffix and reverse-prompt parameters:

```bash
./llama-cli -m your_model.gguf -p "You are a helpful assistant" -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
```

### Web server

[llama.cpp web server](./examples/server/README.md) is a lightweight [OpenAI API](https://github.com/openai/openai-openapi) compatible HTTP server that can be used to serve local models and easily connect them to existing clients.

Example usage:

```bash
./llama-server -m your_model.gguf --port 8080

# Basic web UI can be accessed via browser: http://localhost:8080
# Chat completion endpoint: http://localhost:8080/v1/chat/completions
```

### Interactive mode

> [!NOTE]
> If you prefer basic usage, please consider using conversation mode instead of interactive mode

In this mode, you can always interrupt generation by pressing Ctrl+C and entering one or more lines of text, which will be converted into tokens and appended to the current context. You can also specify a *reverse prompt* with the parameter `-r "reverse prompt string"`. This will result in user input being prompted whenever the exact tokens of the reverse prompt string are encountered in the generation. A typical use is to use a prompt that makes LLaMA emulate a chat between multiple users, say Alice and Bob, and pass `-r "Alice:"`.

Here is an example of a few-shot interaction, invoked with the command

```bash
# default arguments using a 7B model
./examples/chat.sh

# advanced chat with a 13B model
./examples/chat-13B.sh

# custom arguments using a 13B model
./llama-cli -m ./models/13B/ggml-model-q4_0.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

Note the use of `--color` to distinguish between user input and generated text. Other parameters are explained in more detail in the [README](examples/main/README.md) for the `llama-cli` example program.

![image](https://user-images.githubusercontent.com/1991296/224575029-2af3c7dc-5a65-4f64-a6bb-517a532aea38.png)

### Persistent Interaction

The prompt, user inputs, and model generations can be saved and resumed across calls to `./llama-cli` by leveraging `--prompt-cache` and `--prompt-cache-all`. The `./examples/chat-persistent.sh` script demonstrates this with support for long-running, resumable chat sessions. To use this example, you must provide a file to cache the initial chat prompt and a directory to save the chat session, and may optionally provide the same variables as `chat-13B.sh`. The same prompt cache can be reused for new chat sessions. Note that both prompt cache and chat directory are tied to the initial prompt (`PROMPT_TEMPLATE`) and the model file.

```bash
# Start a new chat
PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/default ./examples/chat-persistent.sh

# Resume that chat
PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/default ./examples/chat-persistent.sh

# Start a different chat with the same prompt/model
PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/another ./examples/chat-persistent.sh

# Different prompt cache for different prompt/model
PROMPT_TEMPLATE=./prompts/chat-with-bob.txt PROMPT_CACHE_FILE=bob.prompt.bin \
    CHAT_SAVE_DIR=./chat/bob ./examples/chat-persistent.sh
```

### Constrained output with grammars

`llama.cpp` supports grammars to constrain model output. For example, you can force the model to output JSON only:

```bash
./llama-cli -m ./models/13B/ggml-model-q4_0.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'
```

The `grammars/` folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](./grammars/README.md).

For authoring more complex JSON grammars, you can also check out https://grammar.intrinsiclabs.ai/, a browser app that lets you write TypeScript interfaces which it compiles to GBNF grammars that you can save for local use. Note that the app is built and maintained by members of the community, please file any issues or FRs on [its repo](http://github.com/intrinsiclabsai/gbnfgen) and not this one.

## Build

Please refer to [Build llama.cpp locally](./docs/build.md)

## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](./docs/build.md#metal-build) | Apple Silicon |
| [BLAS](./docs/build.md#blas-build) | All |
| [BLIS](./docs/backend/BLIS.md) | All |
| [SYCL](./docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](./docs/build.md#musa) | Moore Threads GPU |
| [CUDA](./docs/build.md#cuda) | Nvidia GPU |
| [hipBLAS](./docs/build.md#hipblas) | AMD GPU |
| [Vulkan](./docs/build.md#vulkan) | GPU |
| [CANN](./docs/build.md#cann) | Ascend NPU |

## Tools

### Prepare and Quantize

> [!NOTE]
> You can use the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space on Hugging Face to quantise your model weights without any setup too. It is synced from `llama.cpp` main every 6 hours.

To obtain the official LLaMA 2 weights please see the <a href="#obtaining-and-using-the-facebook-llama-2-model">Obtaining and using the Facebook LLaMA 2 model</a> section. There is also a large selection of pre-quantized `gguf` models available on Hugging Face.

Note: `convert.py` has been moved to `examples/convert_legacy_llama.py` and shouldn't be used for anything other than `Llama/Llama2/Mistral` models and their derivatives.
It does not support LLaMA 3, you can use `convert_hf_to_gguf.py` with LLaMA 3 downloaded from Hugging Face.

To learn more about quantizing model, [read this documentation](./examples/quantize/README.md)

### Perplexity (measuring model quality)

You can use the `perplexity` example to measure perplexity over a given prompt (lower perplexity is better).
For more information, see [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity).

To learn more how to measure perplexity using llama.cpp, [read this documentation](./examples/perplexity/README.md)

## Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues and PRs is very appreciated!
- See [good first issues](https://github.com/ggerganov/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggerganov/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentations

- [main (cli)](./examples/main/README.md)
- [server](./examples/server/README.md)
- [jeopardy](./examples/jeopardy/README.md)
- [GBNF grammars](./grammars/README.md)

**Development documentations**

- [How to build](./docs/build.md)
- [Running on Docker](./docs/docker.md)
- [Build on Android](./docs/android.md)
- [Performance troubleshooting](./docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggerganov/llama.cpp/wiki/GGML-Tips-&-Tricks)

**Seminal papers and background on the models**

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)


