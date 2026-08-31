# Tesla V100 (sm_70) — build, deployment and findings

Rig: 2× Tesla V100-SXM2-16GB, `--split-mode tensor`, Qwen3.8-27B-abliterated-Q6_K
(hybrid Gated-DeltaNet + attention, arch `qwen35`), llama-server with MTP
speculative decoding. Everything below was measured on that machine on 2026-08-31
unless a line explicitly says otherwise.

This branch carries two patches. One is a CUDA FlashAttention dispatch fix (§4.2),
the other stops an optional request label from returning HTTP 400 (§4.8). Both are
written to be upstream-submittable and are not V100-specific.

## 0. What this branch is

| branch | contents |
|---|---|
| `master` | pristine upstream `fewtarius/CachyLLama` @ `4ec44dc10` (2026-08-30) |
| `v100` | `master` + one CUDA patch + build/deploy assets + these notes |

The patch is written to be upstream-submittable against `ggml-org/llama.cpp` and is
also kept as a standalone file in `patches/`. It is **not** V100-specific — it fixes
a CUDA dispatch decision that affects every NVIDIA GPU. It is on this branch only
because this is where it was needed and verified.

```
git remote add upstream https://github.com/fewtarius/CachyLLama.git
git fetch upstream && git rebase upstream/master   # on branch v100
```

## 1. Baseline

| | |
|---|---|
| host | Debian 13.3 (trixie), kernel 6.17.9-1-pve |
| CPU / RAM | 2× Xeon Gold 6154 (36c/72t), 125 GiB |
| GPU | 2× Tesla V100-SXM2-16GB, **sm_70**, driver 580.95.05 |
| CUDA | 12.8.1 in the build/runtime image (driver exposes 13.0) |
| Docker | 29.4.2, `nvidia` runtime registered |
| model | `Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf` + `mmproj-F16.gguf` |

Model shape that matters for every calculation below, from the GGUF metadata:

```
qwen35.block_count            = 65      # 64 model layers + 1 MTP (nextn) layer
qwen35.full_attention_interval= 4       # -> 16 full-attention layers, 49 GDN layers
qwen35.attention.head_count_kv= 4
qwen35.attention.key_length   = 256
qwen35.attention.value_length = 256
qwen35.context_length         = 262144  # architectural cap
```

So the KV cache costs `17 × (bytes_per_element_K + bytes_per_element_V)` KiB per
token: 16 attention layers plus the 1-layer MTP draft context.

## 2. Build — sm_70 only

`.devops/cuda.Dockerfile` **fails** on this host: it builds the embedded web UI and
`npm ci` dies in the `@tailwindcss/oxide` postinstall (EPERM under the container's
apparmor/user setup). Use `cuda-server.Dockerfile` (in the repo root on this branch)
instead — same cmake flags as upstream's, no web UI, server + CLI only:

```bash
docker buildx build \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg CUDA_DOCKER_ARCH=70 \
  -f cuda-server.Dockerfile --target server \
  -t cachyllama:server-cuda-$(git rev-parse --short HEAD)-sm70 .
```

`CUDA_DOCKER_ARCH=70` builds **only** sm_70 cubins. Build time ~4.5 min on 72
threads. Verify:

```
$ docker run --rm cachyllama:server-cuda-<sha>-sm70 --version
version: 0.3.0-dev (build 10816, commit 4ec44dc10)
built with GNU 14.2.0 for Linux x86_64
```

`--split-mode tensor` requires flash attention; the server enables it automatically
but `-fa on` is set explicitly in all profiles here.

## 3. Run

Four compose profiles in `deploy/v100/`. They all declare the **same project +
service key**, so `up -d` on any one of them replaces whichever is running — that is
the switch and the rollback, one command each:

```bash
cd deploy/v100
docker compose -f dual-mtp-3.8_original-llamacpp.yml       up -d   # upstream llama.cpp, reference
docker compose -f dual-mtp-3.8_cachyllama.yml              up -d   # production: q8_0 KV, -ub 1024, 150k
docker compose -f dual-mtp-3.8_cachyllama-q5_1-longctx.yml up -d   # q5_1 KV, -ub 512, 205k  (needs the patch)
docker compose -f dual-mtp-3.8_cachyllama-nossd.yml        up -d   # diagnostic: args byte-identical to upstream
```

Override with env: `MODEL_DIR`, `SSD_CACHE_DIR`, `CTX_SIZE`, `UBATCH_SIZE`,
`KV_TYPE`, `KV_TYPE_DRAFT`, `ESTATE_PORT`, `NP`.

Measured, both profiles serving the same model on the same two cards:

| profile | VRAM | prefill | decode @n=192 | context |
|---|---|---|---|---|
| q8_0 / `-ub 1024` | 31 454 MiB | 1035 tok/s | 62.1 tok/s | 150 000 |
| q5_1 / `-ub 512` | 31 434 MiB | 863 tok/s | 57.4 tok/s | **205 000** |

### Gotcha: plain `--gpus all` does not work here

The daemon has the `nvidia` runtime registered but not as default, and the
container-toolkit hook path fails (`ggml_cuda_init: failed to initialize CUDA: OS
call failed or operation not supported on this OS`). For ad-hoc runs use:

```bash
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --security-opt apparmor=unconfined ...
```

The compose files use `deploy.resources.reservations.devices`, which works.

## 4. Findings

### 4.1 The hybrid warm-slot / cache path: an abort, an inverted gate, and the loops

A build from `b9ed083e4` (2026-08-21) died on the **second** request of any
conversation:

```
common/common.cpp:1560: failed to remove sequence 0 with p0=1241, p1=-1
  ggml_abort ← common_memory::seq_rm(int,int,int)        [SIGABRT]
```

Path: `server-context.cpp` truncates the warm slot beyond the reused prefix with
`slot.mem.seq_rm(slot.id, p0, -1)`. On a hybrid GDN model this reaches
`llama_memory_recurrent::seq_rm`, which could only roll back `n_rs_seq` positions and
returned `false` for anything longer (a previous turn's generated tokens).
`common_context_seq_rm` aborts unconditionally on `false`.

`b83d23022` (2026-08-22 — *one day after* that checkout) made
`llama_memory_recurrent::seq_rm` fall through to the regular cell-clearing loop
instead of returning `false`. **That removed the crash without fixing the cause.**
The cell-clearing loop sets `tail_id = -1`, i.e. it *drops* the sequence's recurrent
state, while the attention cache keeps the prefix. The server still believes `p0`
tokens are cached, so the 49 Gated-DeltaNet layers run from a zero state over a long
context while the 16 attention layers see the full history. The model does not crash;
it produces confident, repeating nonsense. A loud failure became a silent one.

#### The actual root cause: an inverted gate

`tools/server/server-context.cpp` guards that truncation with a recovery block —
restore a context checkpoint, else `do_reset` with `n_past = 0`. Upstream:

```c
if (pos_min >= pos_min_thold) {   // ggml-org/llama.cpp
```

CachyLLama since `6af265fa1` (2026-08-20, the direct **parent** of `b9ed083e4`):

```c
if (pos_min <= pos_min_thold) {   // fork
```

`pos_min` is the lowest position the memory still holds for the sequence — on a hybrid
it is `max(attn.pos_min, recr.pos_min)`, so it tracks the *recurrent tail*, which sits
past the LCP whenever the previous turn generated tokens the client does not echo
back (a dropped `<think>` block does it every single turn). `pos_min_thold` is the
largest `pos_min` a checkpoint may have and still cover the LCP. The recovery is
needed exactly when `pos_min` has risen **to or above** the threshold. Inverted, it
fires in the healthy case and is skipped in the broken one, so execution reaches
`slot.mem.seq_rm(slot.id, p0, -1)` with a rollback the recurrent cache cannot perform.

The bound is small: `common_params_speculative::need_n_rs_seq()` returns
`draft.n_max`, so `--spec-draft-n-max 3` means the recurrent cache can roll back
**3 positions**. Any dropped reasoning block blows through that.

The commit message of `6af265fa1` says the intent was to fix
`it->pos_min < pos_min_thold` *inside* the `find_if` below — that predicate is
unchanged and still matches upstream; only the outer gate moved. The stated goal was
to stop "silently forcing full re-prefill on warm slots".

#### Measured, and fixed on this branch

Identical harness (109 208-token context, five turns, only the visible `content`
echoed back so the reasoning block is dropped each turn), temperature 0, SSD cache
cleared before each run — at temperature 0 nothing but the internal state can change
the output:

| build | t0 | t1 | t2 | t3 | t4 | wrong |
|---|---|---|---|---|---|---|
| inverted gate | team-**16** ✗ | 56 | r0 | 144 | **1** ✗ | **2/5** |
| gate restored (`974af40b3`) | **team-18** ✓ | 56 | r0 | 144 | **0** ✓ | **0/5** |

And the concern that motivated the inversion does not materialise: with the gate
restored the same run reports `cache_n = 109207`, `prompt_n = 33` — the checkpoint
ring covers the LCP, so there is **no** full re-prefill.

**Consequences:**

* The operator's `warm-slot-seqrm.patch` was aimed at the right line for the wrong
  reason. Routing the warm path through `llama_memory_seq_rm_attn_only` hides the
  symptom; the gate is the cause.
* Do not run a CachyLLama build between `6af265fa1` and this fix on a hybrid
  (Gated-DeltaNet / Mamba) model. Before `b83d23022` it crashes; after, it degrades
  silently.
* The loops **did** recur on `974af40b3`, so the inverted gate was not their cause.
  The path that is follows below. The two are independent: the gate corrupts while
  truncating a *warm* slot, the SSD restore corrupts on *cold start*.

#### What actually caused the loops: a recurrent state restored at the wrong length

`105889b46` made the cold-start SSD restore accept **partial** LCP matches ("fix cache
hit collapse after agent trims"). The hybrid gate below it had three cases, and case 3
was "partial coverage → cap `n_past` to the LCP". That cap cannot work:

* a recurrent state is one fixed-size tensor, folded over exactly the tokens the
  checkpoint was captured over. It is valid at that one length and nowhere else, and
  there is no representation of "the same state, N tokens earlier";
* `llama_memory_seq_rm_attn_only()` deliberately *preserves* the recurrent R/S data
  (its own comment says so) and clears only attention cells and stale positions.

So after the cap the 49 Gated-DeltaNet layers still carry the state of the **whole**
stored sequence — including the previous turn's generated output — while the server
believes only `ssd_lcp` tokens are cached and prefills the rest. Generation continues
from a state that encodes the text the model just produced, which pulls it back onto
those exact tokens; `--repeat-penalty 1.0` brakes nothing. An agent that trims or
rewrites history produces precisely the partial match this triggers on, every turn.

Measured, both builds, same test — fresh prompt, restart, identical prompt, restart,
then a prompt with a chunk removed after the first ~11k tokens
(`lcp_ratio = ssd_lcp / min(n_tokens, 4096)`, so the ratio is 100 % and the old build
is forced into its cap branch rather than its reject branch):

| run 3 — trimmed prompt, cold start | `974af40b3` | `525620403` |
|---|---|---|
| decision | `partial-coverage: lcp=4096 ssd_n_tokens=27224` **`cap to LCP`** | **`rejecting`**` … partial coverage cannot be truncated` |
| tokens the restored state was captured over | 27 224 | — |
| tokens the server then believed cached (`n_push`) | **4 096** | — |
| `cache_n` / `prompt_n` | 4 096 / 19 724 | 0 / 23 820 |
| run 2 — identical prompt, cold start | `cache_n = 27 223`, `prompt_n = 1` | `cache_n = 27 223`, `prompt_n = 1` |

Fix on this branch: for recurrent/hybrid contexts accept **full coverage or nothing**.
Run 2 shows the headline feature — full state reuse across a container restart — is
untouched; only the unsound path is gone. The cost is bounded: in the trimmed case
4 096 tokens of prefill that were previously "reused" wrongly, and in the ordinary
agentic case, where history only grows, coverage is full and nothing changes.

The same construction sat in the system-prompt cache. `load_prefix()` matches a stored
entry on its first 64 tokens only; the code then set `n_past` to *this* task's
boundary, arguing the "state drift is bounded by the small divergent region at the
boundary" — not a property a recurrent state has. Disabled for hybrid contexts, left
for dense ones where the same construction is also wrong but degrades instead of
wedging.

**Not proven:** the loop itself was never reproduced synthetically — six deliberate
attempts stayed clean, because a hand-written harness appends cleanly and never
produces a partial match. What is proven is that this path fires on exactly the input
an agent loop produces, and what it hands the model when it does. Upstream has no SSD
cache tier at all, which is consistent with two weeks of the same client on `b10236`
without a single occurrence.

### 4.2 FlashAttention + quantized KV under `--split-mode tensor`

Measured on 2× V100, `-sm tensor -ts 1,1 -fa on`, before the patch:

| `--cache-type-k/v` | result |
|---|---|
| `f16`, `bf16`, `q8_0`, `q4_0` | works |
| `q4_1`, `q5_0`, `q5_1`, `iq4_nl` | `ggml-backend-meta.cpp:537: GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) failed` |
| any `K != V` combination | same abort |

There is **no `q6` KV cache type** in llama.cpp at all — `kv_cache_types`
(`common/arg.cpp`) is `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`. Q6_K is
a weight-only super-block quant; `q6_0` exists only in `ik_llama.cpp`.

Root cause: `ggml_cuda_fattn_kv_type_supported()` gated **every** FA kernel on the
availability of a per-type *vector* kernel instance, and a default build
(`GGML_CUDA_FA_ALL_QUANTS=OFF`) compiles only four: `f16-f16`, `q4_0-q4_0`,
`q8_0-q8_0`, `bf16-bf16`. Everything else was reported unsupported, so the scheduler
moved the `FLASH_ATTN_EXT` node to the **CPU** backend. With `-sm none/layer` that is
merely catastrophic for speed; with `-sm tensor` the meta backend cannot derive a
split state for a node it does not own, and aborts.

But the **tile** and **MMA_F16** kernels never read the KV type directly:
`launch_fattn()` dequantizes K and V to F16 first (`need_f16_K`/`need_f16_V`) — which
is already how `q8_0`/`q4_0` are served on Volta. Only the vec kernels are per-type
templates. The patch splits the predicate accordingly, adds the two small pieces
`iq4_nl` was missing (a non-contiguous dequant kernel, and the `iq4_nl → f32` copy the
quantized K-shift path needs), and fixes `docs/multi-gpu.md`, which still claimed
quantized KV is unsupported in this mode and documented an error string that no longer
exists.

No new template instances are compiled, so build time and binary size are unchanged,
and kernel selection for `f16/bf16/q4_0/q8_0` is bit-for-bit identical.

After the patch, all of them start and answer correctly:

| `--cache-type-k/v` | before | after |
|---|---|---|
| `f16` / `q8_0` / `q4_0` | ok | ok |
| `q4_1` / `q5_0` / `q5_1` / `iq4_nl` | abort | **ok** |
| `q8_0` / `q5_1` (asymmetric) | abort | **ok** |

Prior art, for reference: issue
[#27116](https://github.com/ggml-org/llama.cpp/issues/27116) is exactly this abort;
[#23567](https://github.com/ggml-org/llama.cpp/issues/23567) and
[#21788](https://github.com/ggml-org/llama.cpp/issues/21788) are older, now-stale
feature requests; PR
[#27248](https://github.com/ggml-org/llama.cpp/pull/27248) (art-den, open, no review)
solves it differently and far more invasively, by adding graph-level dequantization
state to `llama_context`.

Cost of the newly enabled types: they use tile/MMA with an on-the-fly F16 conversion
instead of a native vec kernel. On Volta this is nearly free, because the vec kernel is
only selected when `Q->ne[1] * gqa_ratio_eff <= 2` and the MTP draft already makes
`Q->ne[1] = 4`. Build with `-DGGML_CUDA_FA_ALL_QUANTS=ON` to get native vec kernels
for all types at the price of a longer compile.

### 4.3 KV-cache quantization: quality

Researched, not measured here. Two things make most of the public record only
partly applicable: (a) since `ggml-org/llama.cpp` PR
[#21038](https://github.com/ggml-org/llama.cpp/pull/21038) (merged 2026-04-01) a
Hadamard activation rotation is applied **by default** to each quantized KV side
when `head_dim % 64 == 0` — this model has `key_length = value_length = 256`, so it
is active, and every pre-2026-04 benchmark describes a code path we do not run;
(b) nothing public measures llama.cpp cache types above **128k**, and we run
150k-205k.

#### Upstream, maintainer-produced

`llama-perplexity --kl-divergence`, Qwen3.5-9B, wiki.test.raw, **with** rotation
(AesSedai in PR #21038, [comment](https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570)):

| K / V | mean KLD | × vs q8_0/q8_0 |
|---|---|---|
| f16 / f16 | 0.000782 | 0.98 |
| **q8_0 / q8_0** | **0.000799** | 1.00 |
| q8_0 / q5_0 | 0.001229 | 1.54 |
| **q8_0 / q5_1** | **0.001300** | **1.63** |
| **q5_1 / q5_1** | **0.001839** | **2.30** |
| q8_0 / q4_0 | 0.002568 | 3.21 |
| q4_0 / q4_0 | 0.004782 | 5.98 |

Perplexity separates these by <0.03 % — ggerganov's own guidance in that PR is
*"it seems important to track the KLD rather than PPL"*.

The **only task-level benchmark upstream**, ggerganov on gpt-oss-20b, AIME25 ×8
([comment](https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4150413357)),
with rotation: F16 **37.9 %**, Q8_0 **37.1 %**, Q5_1 **32.5 %**, Q5_0 32.5 %,
Q4_1 28.3 %, Q4_0 21.7 %. So on a multi-step reasoning task — the closest available
proxy for agentic work — `q8_0` is level with F16 while `q5_1` costs **4.6 points**.
Note also how much the rotation itself is worth: Q4_0 was **2.0 %** without it.

`docs/function-calling.md` carries the only quality recommendation in the repo:
*"Beware of extreme KV quantizations (e.g. `-ctk q4_0`), they can substantially
degrade the model's tool calling performance."* No data is cited upstream for it.

#### Long context, closest model family to ours

Qwen3.6-27B at **128k**, KL vs a `bf16` cache, IQ4_XS weights
([anbeeld.com](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context)) —
the only public long-context grid using llama.cpp cache types:

| K / V | % of bf16 KV | mean KLD | 99.9 % tail precision | same-top-token |
|---|---|---|---|---|
| bf16 / bf16 | 100.0 | 0.000000 | 100.00 % | 99.995 % |
| **q8_0 / q8_0** | 53.1 | 0.000482 | **98.52 %** | 98.950 % |
| **q8_0 / q5_1** | 45.3 | 0.000651 | **98.13 %** | 98.779 % |
| **q5_1 / q5_1** | 37.5 | 0.000827 | **97.70 %** | 98.603 % |
| q4_0 / q4_0 | 28.1 | 0.002259 | 94.34 % | 97.793 % |

A 50k-context sweep over a 230 MB pure-Python corpus on the same model family
([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1uq0fpe/)) adds the
useful detail that **K at q8_0 is essentially free**: `f16/q8_0` 0.005399 vs
`q8_0/q8_0` 0.005410, and `q8_0/q5_1` 0.007360. A 16k coding-corpus sweep
([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1tlwjsl/)) concludes
*"q4_0 cache … is basically never worth it. Use at least q5_1."*

#### K vs V asymmetry — the famous 6.6× does not replicate

The claim originates in PR #7412 (2024): at equal bit budget, quantizing only K vs
only V cost 1.4× at q8_0 rising to **6.6×** at q4_0. On modern rotated builds of
Qwen3.6-27B the same comparison is **1.18-1.40×**. The structural reason K is the
weaker side is real — K has persistent per-channel outliers while llama.cpp
quantizes per token ([KVQuant](https://arxiv.org/pdf/2401.18079) App. G,
[KIVI](https://arxiv.org/pdf/2402.02750) Tbl. 2) — but sizing a cache off the 6.6×
figure over-fits one unlabelled 2024 run. What *does* replicate in all three
datasets that measured both: **`K=q8_0 / V=q5_1` costs only 29-49 % of symmetric
`q5_1`'s incremental penalty for 21 % more KV memory** (7.25 vs 6.00 bpv).

#### What this means for this rig

* Keep **q8_0/q8_0** while 150k is enough. It is level with f16 on every metric
  above, and this is a tool-calling workload — the one category-resolved study
  ([localbench](https://localbench.substack.com/p/kv-cache-quantization-benchmark))
  finds coding itself near-undamaged but ranks **tool calling and long-document
  retrieval as the two exposed surfaces**.
* If context must grow, prefer **`K=q8_0 / V=q5_1`** (§4.4: ~164k) over symmetric
  `q5_1` (~205k). It buys most of the memory at a third to a half of the quality
  cost, and no source ranks it worse than symmetric `q5_1`. The shipped
  long-context profile exposes `KV_TYPE_K` / `KV_TYPE_V` separately for exactly
  this.
* Do not go to `q4_0`/`q4_1` for this workload. Upstream's own docs warn about it
  for tool calling, AIME25 drops 15 points, and
  [TriAxialKV](https://arxiv.org/pdf/2605.17170) Tbl. 2 measures 4-bit KV costing
  5.6-7.1 points of BFCL function-calling accuracy on Qwen3-14B/32B.
* **Model family dominates the bit choice.** On Gemma 4 a `q8_0` cache alone is
  already worse (KL 0.108 dense / 0.377 MoE) than Qwen's `q4_0`. None of this
  transfers to a different model — re-validate if the served model changes.
* Nobody has measured any of this at 150k-205k, and the direction of
  length-scaling is agreed but unquantified. Treat the 205k profile as unproven on
  quality, and if long-context output ever collapses, rule out a kernel/accumulation
  bug before blaming the codec — vLLM saw 128k needle accuracy go 91 % → 13 % → 89 %
  from an FP32-accumulation bug alone.

### 4.4 VRAM budget — what actually scales with context

Four measurements (q8_0/q8_0, MTP on) pin down a four-term model:

| measurement | VRAM total |
|---|---|
| `-c 150000 -ub 1024` | 31 454 MiB |
| `-c 150000 -ub 512` | 30 546 MiB |
| `-c 150000 -ub 256` | 30 090 MiB |
| `-c 180000 -ub 512` | 32 202 MiB |

`V = F + a·n_ctx + b·n_ubatch + c·n_ctx·n_ubatch`, which decomposes as:

| component | MiB @ 150k/ub1024 | KiB/token | scales with |
|---|---|---|---|
| KV cache, 17 layer-equivalents | 5 292 | 36.12 | `n_ctx` × KV type |
| KQ mask, device copies | 1 184 | 8.08 | `n_ctx` × **`n_ubatch`** |
| FA F16 scratch + other ctx-only buffers | 2 401 | 16.39 | `n_ctx` |
| activations / FFN intermediates | 632 | — | `n_ubatch` |
| weights + GDN recurrent state + CUDA contexts | 21 945 | — | fixed |
| **total** | **31 454** | | = measurement, exactly |

* **KQ mask** — `self_kq_mask` is `[n_kv, n_ubatch]`. With `-fa on` it is allocated
  directly as F16 and `self_kq_mask_cnv` aliases it (`llama-graph.cpp:39`), so 2 B per
  element. Measured 8.08 B per (ctx token × ubatch slot) = 2 B × **2 contexts**
  (target + MTP draft) × **2 devices** (MIRRORED under `-sm tensor`). It is the only
  term in the budget that scales with a *product*, hence 1.2 GiB at 150k × 1024 and
  ~1 MiB during decode. The buffer is reserved at startup for the worst case, so it is
  resident even though decode uses one row of it.
* **FA F16 scratch** — `launch_fattn()` dequantizes K and V into space appended to the
  FA node's dst (`fattn-common.cuh:68-80`). One copy is
  `n_head_kv`(4) × (256+256) × 2 B = 4.0 KiB/token; the measured 16.39 is ~4 live
  buffers. Being F16 by construction, **it does not shrink when you quantize the KV
  cache harder** — this is what makes returns diminish below q5_1.
* **MTP draft context** — dropping `--spec-type draft-mtp` frees **3 252 MiB** at 150k,
  of which only 312 MiB is the draft KV; the rest is a second full `llama_context` with
  its own `n_ctx`-sized mask and FA buffers for a single layer. That would allow
  `-c ≈ 222 000` at q8_0 (measured: `-c 220000` no-MTP = 31 354 MiB) but decode falls
  from 54.7 to **37.2 tok/s** @ n=192. Keep MTP.

Maximum context at the same VRAM headroom the working 150k/q8_0 config leaves.
The model reproduces that config exactly (150 016) as a self-check:

| `-ub` | K / V | KiB/token | max `-c` |
|---|---|---|---|
| 1024 | q8_0 / q8_0 | 60.6 | 150 000 ✔ verified |
| 1024 | q8_0 / q5_1 | 55.3 | 164 000 (176 000 verified to start) |
| 1024 | q5_1 / q5_1 | 50.0 | 182 000 (195 000 verified to start) |
| 1024 | q4_0 / q4_0 | 43.6 | 208 000 |
| 512 | q8_0 / q8_0 | 56.6 | 166 000 (180 000 verified to start) |
| 512 | q5_1 / q5_1 | 45.9 | **205 000 ✔ verified, shipped as a profile** |
| 256 | q5_1 / q5_1 | 43.9 | 218 000 |
| 256 | q4_0 / q4_0 | 37.5 | 255 000 |

`-ub` is therefore a **memory** lever, and it costs prefill only:

| `-ub` | VRAM @150k | prefill (9040-token prompt) | decode @n=192 |
|---|---|---|---|
| 1024 | 31 454 MiB | 1035 tok/s | 62.1 |
| 512 | 30 546 MiB | 943 (−9 %) | 61.5 |
| 256 | 30 090 MiB | 764 (−26 %) | 61.1 |

Decode is flat across a 4× range because it submits 1 token (≈4 with the MTP draft),
orders of magnitude below any `-ub`. Prefill is the matrix-matrix regime: a narrower
ubatch means more graph executions over the same tokens, each re-reading the full
weight set with a smaller GEMM M dimension.

### 4.5 Reported decode tok/s is dominated by a fixed per-request cost

A single-length benchmark says CachyLLama loses ~23 % decode against upstream. That is
an artefact. Sweeping `n_predict` and fitting `predicted_ms = n·t + C`:

| build | ms/token | asymptotic tok/s | fixed ms/request |
|---|---|---|---|
| upstream llama.cpp b10236 | 12.512 | 79.9 | 241 |
| this fork, `--cache-ssd` off | 12.530 | 79.8 | 619 |
| this fork, `--cache-ssd` on | 12.465 | 80.2 | 1076 |

**The forward pass is identical** (within 0.5 %). The whole difference is a fixed cost
charged to `predicted_ms`, so the apparent loss shrinks with generation length:
−51 % at 48 tokens, −24 % at 192, −15 % at 361, ±0 asymptotically. For an agent
workload of short tool-call turns this is the worst case.

Source: `deferred_create_final_checkpoint()` in `tools/server/server-context.cpp`,
called from the generation loop right after the first token is sent. Per request, for
`ctx_tgt` **and** `ctx_dft`: read the full live KV state GPU→host, strip generated
tokens with `seq_rm_attn_only`, read the stripped state GPU→host **again** into the
checkpoint, then write the saved state back host→GPU — up to six large transfers
(~670 MiB per the code's own comment), plus the SSD write when that tier is on. The
header comment says it "does not block the first token", which is true; it blocks
tokens 2…n.

Not the cause, checked: `fsync` (`--cache-ssd-no-fsync` moved it by 36 ms, noise) and
MoE-residency / expert tracking (`llama-context.cpp:2123-2198` is gated on
`expert_tracking_enabled`, which llama-server never sets).

Snapshotting at the prompt boundary instead of after the first token, or moving the
work off the generation thread, would remove this without losing the warm-restart
capability.

### 4.6 The cache is three tiers, two of them RAM; `-np 1` kills the fourth

The server says so itself at startup:

```
--cache-ram is a no-op with a single slot (the prompt cache cannot accumulate state…)
```

`f658fc5af` deliberately skips the host-memory prompt-cache round-trip when
`n_parallel <= 1`. Measured on an A → B → A conversation switch with ~9k-token
prefixes, turn "A2":

| build | wall | tokens reused |
|---|---|---|
| upstream llama.cpp b10236 (RAM cache, default 8192 MiB) | 1.80 s | 9036 / 9040 |
| this fork, `--cache-ssd` on | 2.53 s | 9039 / 9040 |
| this fork, `--cache-ssd` off | 9.54 s | **0** |

So `--cache-ssd` is load-bearing here, not an optimisation — without it a single-slot
server gets no cross-conversation reuse at all. Note also that upstream's plain RAM
cache is *faster* at this than the SSD tier, so the fork's advantage at `-np 1` is
persistence and capacity, not latency.

The SSD checkpoint compat hash encodes the KV type (`type_k=8` q8_0 →
`a6e64a7b2a475306`, `type_k=7` q5_1 → `33e4eb71fcdb22b6`), so profiles with different
KV types cannot share checkpoints. Give each profile its own `SSD_CACHE_DIR`. The
on-disk format also changed in `59da9e100` ("kv-ssd: v4 format"), so caches written by
pre-2026-08-22 builds are stale.

#### Tier layout

`--cache-ssd` is not a disk cache. It is a tiered store with two RAM levels in front
of the disk, and the deployed profile sizes them explicitly:

| tier | where | budget | window flag |
|---|---|---|---|
| hot | host RAM | `--cache-ssd-hot-ram 8192` MiB | `--cache-ssd-hot-window 150000` tok — always keep |
| warm | host RAM | `--cache-ssd-warm-ram 8192` MiB | `--cache-ssd-warm-window` (default 32768 tok) — keep in RAM |
| cold | disk | `--cache-ssd-cold-maxsize 356000` MiB | — |
| (`--cache-ram`) | host RAM | upstream's prompt cache — **set to `0` here** | dead at `-np 1`, see below |

The deployed profiles set `--cache-ram 0`. It reads like a second, competing host-RAM
cache in front of the fork's own two tiers, but `8192` was only restating the upstream
default (`common/common.h:632`), and at `-np 1` it is inert either way — the fork's own
comment says the round-trip is skipped "but the memory is still reserved"
(`server-context.cpp:1467`), and the server emits `SRV_WRN: --cache-ram is a no-op with
a single slot`. Measured container RSS at startup, before any request:

| `--cache-ram` | RSS after load | reuse on an A → B → A switch |
|---|---|---|
| 8192 | 2.437 GiB | `cache_n = 9036` (§4.6 table above) |
| 0 | 2.439 GiB | `cache_n = 5227`, `prompt_n = 1`, server-side `total time = 2233 ms` |

So "reserved" means the budget, not an allocation: it costs zero bytes. Setting it to
`0` buys no memory — it removes the warning and one dead code path. The single side
effect is that `--cache-idle-slots` turns itself off (`--cache-idle-slots requires
--cache-ram, disabling`), which is free here: that feature publishes idle slots through
the same save path that `-np 1` already skips.

Do not set it to `0` on a `--parallel > 1` deployment — there the round-trip is real and
it is the only cross-*task* prompt reuse upstream has.

Startup confirms the two RAM budgets, and the per-turn line shows them filling:

```
srv load_model: SSD cache enabled: path=/cache/kv, hot=8192 MiB, warm=8192 MiB
SSD cache: turn 96 complete (hot=4434 MiB warm=8428 MiB cold=0 checkpoints=7)
```

So a hit is served from RAM whenever the conversation is recent; the disk is the
overflow and the persistence layer. Budget accordingly — the two tiers are 16 GiB of
the host's 125 GiB, *on top of* the ~21 GiB the cold tier currently holds on disk.

#### Reboot survival

The cold tier survives a restart and a reboot. It is a plain file tree
(`ckpt-N.bin`, 1–2.5 GiB each) on a persistent filesystem, bind-mounted in:

```
bind /nvme/llm/cachyllama-cache -> /cache        # zfs, pool "nvme"
```

Metal-proven across a container restart: `docker compose down && up`, then the same
prompt → `cache_n = 27223`, `prompt_n = 1` (§4.1). The RAM tiers are of course lost
and repopulate from the cold tier, so the first hit after a boot is a disk read.

One rig-specific caveat, and it explains an earlier measurement: this pool runs
**`sync=disabled`**. The server's fsync on checkpoint writes is therefore ignored by
ZFS — which is why `--cache-ssd-no-fsync` measured as a rounding error (1040 vs
1076 ms fixed cost, §4.5): the cost is the GPU→host state copy plus the write, never
the sync. After an *unclean* stop the pool rolls back to the last committed txg
(≤ `zfs_txg_timeout`, 5 s by default), so the newest checkpoints can be missing.
That degrades to a miss, not to corruption: the loader validates a header checksum
(`kv-ssd-cache.cpp:697`), the `model_identity`, the format version, and reads the
payload with `pread_all()`, so a rolled-back or truncated file fails the read and is
treated as a clean miss. A clean `reboot` loses nothing.

### 4.7 Other things worth knowing

* `ghcr.io/ggml-org/llama.cpp:server-cuda` at build **10524** (`9ee9fc04c`) **does not
  start** on this rig with `-sm tensor`:
  `ggml_backend_cuda_buffer_init_tensor: cudaMemset(...): CUDA error: invalid argument`.
  Build 10236 works. Not investigated further; it means you cannot casually pick a
  recent upstream container as an A/B reference here.
* `--split-mode tensor` disables backend sampling (`backend sampling not supported with
  SPLIT_MODE_TENSOR; using CPU`) and `llama_params_fit` (`-fit` must be `off`).
* The `mmproj` vision projector is loaded with `--no-mmproj-offload`, so it costs host
  RAM, not VRAM.

### 4.8 `llama_user_id` returned HTTP 400 on ordinary requests

Switching a client from upstream llama.cpp to this fork can break it outright:

```
got exception: {"error":{"code":400,"message":"llama_user_id must match
^[a-zA-Z0-9\\-_]+$ (empty = anonymous)","type":"invalid_request_error"}}
```

`server_task::validate_user_id()` threw `std::invalid_argument` for any id outside
`^[a-zA-Z0-9\-_]+$`, and the two call sites in `server-chat.cpp` call it while
assembling the OAI body, outside any handler-level try/catch. Any client that sends
an e-mail address, a UUID with dots, or a `tenant:42` style label gets a hard 400 on
a perfectly well-formed completion. The Anthropic path is the easiest way to hit it,
because it promotes `metadata.user_id` verbatim — a field Anthropic documents as
free-form.

The character set was never a safety boundary. The id is hashed with
`sha256_namespace_key()` before it reaches a cache key or the filesystem, and the
on-disk path is `<base>/u/<16 hex digits of that hash>`
(`server_context_page_manager::get_or_create_user_cache`), so no part of the raw
string can escape into a path.

Fixed on this branch by sanitizing instead of throwing: truncate ids longer than
512 bytes, replace control characters so they cannot corrupt a log line, pass
everything else through unchanged. Per-user routing stays injective for any
realistic identifier, and an optional label can no longer fail a completion.

#### The deeper problem: the client decided who gets a cache bucket

Not returning 400 is only half of it. `task.user_id` also *routes* everything, in
`server-context.cpp`:

* the prompt/KV cache bucket — `<cache>/u/<sha256>` instead of `_anonymous`,
* the SSD checkpoint bucket used by `store_checkpoint_with_tokens()` and
  `find_and_load_checkpoint()`,
* slot reuse: a slot owned by a different id is **not** reused (`get_available_slot`),
* slot affinity, and the `--max-concurrent-per-user` cap.

So the value the *client* sends partitions the *server's* prompt cache. A frontend
that puts a session id, a conversation id or a request id in that field — a normal
thing for a chat client to do — lands in a fresh bucket on every request and gets
**zero cross-request prompt reuse**, on a server whose entire reason to exist is
cross-request prompt reuse. The README calls the feature opt-in; in practice it was
opted into by whoever wrote the client.

This branch gates it on a new server flag:

```
--user-isolation            (env LLAMA_ARG_USER_ISOLATION, default OFF)
```

Off (the default): the field is ignored entirely, every request shares the anonymous
bucket, and the prompt cache works across requests unconditionally — which is what a
single-tenant deployment wants. On: the previous multi-tenant behaviour, unchanged.

If you are running an unpatched build and hit either problem, the only workaround is
to stop the client from sending the field at all (`metadata.user_id` on the Anthropic
endpoint, `llama_user_id` or `extra_body` on the OpenAI endpoints).

### 4.9 Observation: rapidly alternating `llama_user_id` values and prefix reuse

Once, immediately after six back-to-back requests each carrying a *different*
`llama_user_id`, a trivial prompt ("Reply with exactly: OK") produced 600 tokens of
fluent but unrelated text (npm `package-lock` fragments) instead of the answer. The
server log around it shows the slot reusing a short prefix from an unrelated
conversation:

```
W slot operator(): id 0 | task 31 | need to evaluate at least 1 token for each active slot (n_past = 15…)
W slot operator(): id 0 | task 31 | n_past was set to 14
I slot get_availabl: user_id check: task=06a1e666 slot=a3a4a372, conv_hash match=1, same…
```

On a hybrid GDN model, reusing a prefix whose recurrent state belongs to a different
conversation is exactly the failure mode that produces confident nonsense — the
attention cells match but the recurrent state does not.

**Not reproducible** in three deliberate attempts afterwards (warm the slot with an
unrelated conversation, then ask the trivial question: 3/3 correct), and not caused by
either patch on this branch: decode throughput was identical to the unpatched build
(54.4 vs 54.68 tok/s on the same config) and the same payloads return correct output.
Recorded because a client that alternates user identities per request is exactly the
workload that creates many per-user cache buckets, and that is the situation in which
it appeared. With `--user-isolation` off (the default on this branch, §4.8) those
buckets are never created, so this configuration cannot arise from client input any
more. If you do see it, capture the log with `-lv 4` and check the `user_id check` /
`n_past was set to` lines.

## 5. Evidence tiers

Metal-proven on 2× V100 (started, served requests, numbers taken from the server's own
`timings` and `nvidia-smi`): everything in §2, §3, §4.1, §4.2, §4.4, §4.5, §4.6, §4.8.

Desk-proven (read from the code, not exercised): the `iq4_nl → f32` copy path added by
the patch is only reached by the quantized K-shift (`build_rope_shift`), which requires
the context to actually fill up — that never happened in testing.

Not measured here at all: long-context quality of any KV quantization (see §4.3 for
published data), soak behaviour of the 205k profile under sustained full-length
prefills, and multi-slot (`-np > 1`) behaviour of any of this.
