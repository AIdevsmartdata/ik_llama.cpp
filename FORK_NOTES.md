# Fork notes — `AIdevsmartdata/ik_llama.cpp`

This is a fork of [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) maintained by [@AIdevsmartdata](https://github.com/AIdevsmartdata) (Kévin Rémondière). It tracks Iwan Kawrakow's upstream and adds a backport of Mamba-2 / Nemotron-H MoE support that does not yet exist in upstream.

## Why this fork exists

`ik_llama.cpp` has the strongest CPU + iqk SIMD matmul + sm_120 (Blackwell) CUDA performance of any `llama.cpp` derivative we tested, plus first-class BitNet, custom IQK quants, MTP support, and the unique combination of fused MoE up/gate + grouped expert routing. But as of March 2026 it does not yet support Mamba-2 or hybrid SSM architectures (Mamba-2, Nemotron-H, Granite 4.0 hybrid, Falcon-H1, Bamba, Hymba, Codestral Mamba, Zamba2, Jamba, …).

We backported the Mamba-2 / Nemotron-H MoE work from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) into this fork so that we can run hybrid SSM models with **all the ik_llama performance optimizations** (iqk matmul, sm_120 CUDA, custom quants, fused MoE), instead of having to fall back to stock `llama.cpp` and lose them.

## Upstream pull request

The backport is offered to upstream as

> **[ikawrakow/ik_llama.cpp#1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593)** — *Mamba-2 + Nemotron-H MoE backport (Phase 3.x)*

12 commits, ~1550 added / 173 deleted lines, 18 files. Phased so each step is independently reviewable.

If you want a clean view of the backport, check out the dedicated branch:

```bash
git checkout mamba2-nemotron-h-backport
```

## What the backport adds

| Phase | What | Files |
|---|---|---|
| **1** | New arch tags `LLM_ARCH_MAMBA2`, `LLM_ARCH_NEMOTRON_H_MOE` | `src/llama-arch.{h,cpp}` |
| **2** | Tensor allocation + hparams loading for both archs (`create_mamba2_tensors`, `create_nemotron_h_moe_tensors`) | `src/llama-load-tensors.cpp`, `src/llama-hparams.{h,cpp}` |
| **2** | Recurrent state size accessors split out (`n_embd_r/s`); first-class `use_qnext_state_layout` flag | `src/llama-hparams.{h,cpp}`, `src/llama.cpp` |
| **3.1** | Defensive SSM bounds + cache layout audit | `src/llama-hparams.cpp`, `src/llama.cpp` |
| **3.2** | Port upstream unified `ggml_ssm_scan` op (Mamba-1 + Mamba-2) and its CUDA backend | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/ssm-scan.{cu,cuh}`, `ggml/include/ggml.h` |
| **3.3** | New `build_mamba2_layer` (mirrors upstream `mamba-base.cpp` line-for-line) and `build_nemotron_h_moe` graph builders dispatched per layer (Mamba-2 SSM mixer / GQA attention / gateless RELU² MoE FFN) | `src/llama-build-context.{h,cpp}` |
| **3.3** | Fix `inp_ssm_ids` to read recurrent slot 0 (not `kv_self.head` which is the attention head index for hybrids) | `src/llama.cpp` |
| **3.3** | Fix the `eval-callback` printer to compute its sum over the whole tensor instead of the truncated head/tail slice (was masking real numerical divergence during the debug session) | `examples/eval-callback/eval-callback.cpp` |
| **3.4** | Fix `llm_build_ffn` `LLM_FFN_PAR` mul to be guarded on `gate != nullptr` — Nemotron-H's gateless `relu²(up_proj)` shared expert was computing `relu²(up) * up` instead of `relu²(up)`. Long-standing latent bug, never triggered before because every prior caller of `LLM_FFN_PAR` happened to also pass a gate (SwiGLU style). | `src/llama-build-context.cpp` |
| **API compat** | Catch up with current upstream API drift (`n_embd_head_k` per-layer accessors after Gemma4, `n_embd_v_s → n_embd_s` rename) | `src/llama-build-context.cpp`, `src/llama-model.cpp` |

## Validation

End-to-end on **NVIDIA Nemotron-3-Nano-30B-A3B** (`unsloth/Nemotron-3-Nano-30B-A3B-GGUF`, hybrid Mamba-2 + GQA + MoE 128 experts top-6) at 2026-04-06 / 07:

| GGUF | Backend | Output |
|---|---|---|
| `Nemotron-3-Nano-30B-A3B-Q4_0.gguf` | sm_120 CUDA via `chimere-server` HTTP | `What is the capital of France? → Paris.` |
| `Nemotron-3-Nano-30B-A3B-Q4_0.gguf` | CPU only (`-ngl 0`) via `llama-cli` | `The capital of France is Paris." Provide exactly that line, no extra text.` |
| `Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf` (Unsloth dynamic) | CPU only via `llama-cli` | `Paris, and the capital of Italy is Rome, but the capital of Ireland is Dublin.` |

Performance: **~45 tok/s** on RTX 5060 Ti (sm_120, Blackwell consumer) with `-ngl 99 --n-cpu-moe 30 -c 2048 -ctk q8_0 -ctv q4_0`.

**Qwen3.5-35B-A3B (the GDN production target of `chimere-server`) is byte-for-byte unchanged** — the cache layout, recurrent state, and forward path of the existing qnext / GDN architecture are not touched. We added a `use_qnext_state_layout` flag and branched the new accessors on it; the old code path remains the default for any model that does not opt into the new accessors.

## Caveats / scope of the backport

These are intentional limits, all clearly documented in the source comments, and all easy to lift in a follow-up patch:

1. **`n_seqs == 1`** for the Mamba-2 graph builder. `build_mamba2_layer` writes the new SSM state back to slot 0 unconditionally (`r_kv_head = 0`) and reads `inp_ssm_ids[s] = s`. Multi-sequence parallel decoding for the new arch is not yet wired (single-slot recurrent cache only). Qwen3.5 GDN multi-seq is unaffected.
2. **State save / restore** (`read/write_kv_cache_data`) is not adapted for the hybrid Nemotron-H path yet. Persistent sessions (`--cache-reuse`) are broken; regular fresh-prompt inference is fine.
3. **Mamba-1 build path is still stubbed.** ik_llama has no production GGUF for Mamba-1 anymore, so we left the legacy `build_mamba()` body in `#if 0` and aborted the dispatch on the Mamba-1 branch.
4. **Phase 3.3 reuses the OLD 4-arg `ggml_ssm_conv` op** for the conv1d step instead of porting upstream's new 2-arg `concat(state, transpose(xBC)) → ssm_conv(conv_x, weight)` rewrite. Numerically identical for `n_seqs=1`, slightly more graph nodes per layer. Can be reworked.

## Downstream user

This fork is the backend of [`AIdevsmartdata/chimere`](https://github.com/AIdevsmartdata/chimere), a Rust HTTP inference server (`chimere-server`) that wraps `libllama.so` via FFI and adds:

- An OpenAI-compatible `/v1/chat/completions` HTTP layer
- C++ fast sampler with DRY + min-p + top-n-sigma + adaptive temperature
- Engram n-gram logit-bias overlay (Cuckoo-filter, ~21 MB of prebuilt domain tables)
- Multi-token prediction (MTP) speculative decoding scheduler
- K-cache Hadamard rotation, fused MoE up/gate
- Multi-agent KV / SSM state save & restore via `llama_state_seq_*`
- Step 7 multi-arch dispatch: the same chimere-server runtime now hosts both Qwen3.5-35B-A3B (full prod stack) and Nemotron-H MoE (libllama-only path) via a closed `AppStateModel { Qwen35, Generic }` enum

`chimere-server` pins this fork via its `LD_LIBRARY_PATH` and validates the backport on every release.

## How to build (sm_120 / Blackwell consumer)

```bash
git clone https://github.com/AIdevsmartdata/ik_llama.cpp.git
cd ik_llama.cpp
git checkout mamba2-nemotron-h-backport       # the clean PR-ready branch
cmake -B build_sm120 \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DGGML_NATIVE=OFF
cmake --build build_sm120 -j
```

Binaries land in `build_sm120/bin/{llama-cli,llama-server,llama-eval-callback,…}`.

Then (for example) on the Nemotron Q4_0:

```bash
./build_sm120/bin/llama-cli \
    -m Nemotron-3-Nano-30B-A3B-Q4_0.gguf \
    -ngl 99 --n-cpu-moe 30 -c 2048 \
    -ctk q8_0 -ctv q4_0 \
    -p "The capital of France is" -n 25
```

## Sync policy

We rebase / merge from `ikawrakow/ik_llama.cpp` periodically. The clean PR branch (`mamba2-nemotron-h-backport`) is rebased on top of the latest upstream main; the local feature branches (`mamba2-backport-phase1` etc.) carry an older base + the user's MTP work and are not intended for upstream merging.

If upstream merges PR #1593 we will delete this fork's local commits and switch to upstream directly.

## License

This fork inherits the upstream MIT license. See [`LICENSE`](./LICENSE).

## Contact

Issues, regressions, or merge questions: open an issue on this fork or comment directly on [PR #1593](https://github.com/ikawrakow/ik_llama.cpp/pull/1593).
