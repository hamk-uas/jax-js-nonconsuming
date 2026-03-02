# WebGPU Performance Benchmark Findings

**Date:** 2026-03-02 (updated 2026-03-02)  
**Hardware:** NVIDIA RTX 4070 Ti SUPER (16 GB) via Razer Core X Chroma eGPU over Thunderbolt 3  
**Driver:** 590.48.01, Vulkan 1.3.275  
**Host OS:** Linux (bare metal)

---

## 1. Test Setup

### Runtimes Compared

| Runtime                            | WebGPU Backend            | GPU Adapter Description                                       | Key Features                                                                 |
| ---------------------------------- | ------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Deno 2.x** (`--unstable-webgpu`) | wgpu-rs (native Vulkan)   | `NVIDIA GeForce RTX 4070 Ti SUPER` (vendor 4318, device 9989) | `shader-f16`, `timestamp-query`, `dual-source-blending` — **no `subgroups`** |
| **Chromium** (Playwright, headed)  | Dawn (via Vulkan → ANGLE) | `nvidia`, architecture `lovelace`                             | `subgroups`, `timestamp-query`, `shader-f16`                                 |

### Vitest Chromium Launch Flags

```ts
// vitest.config.ts (updated — now headless with full Chrome flags)
headless: true,
args: [
  "--no-sandbox",
  "--headless=new",
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--disable-vulkan-surface",
  "--enable-unsafe-webgpu",
],
```

Previously we ran headed with only 3 of 6 flags. The switch to headless with full flags:

- Reduces test startup from ~6s to ~1s
- Maintains NVIDIA Lovelace GPU with `subgroups` feature
- Achieves identical GPU performance to headed mode

### eGPU Latency Context

Thunderbolt 3 provides ~22 Gbps unidirectional (PCIe 3.0 ×4 equivalent). Each CPU↔GPU round-trip
adds ~10-50 μs latency vs ~1 μs native PCIe. This affects:

- `mapAsync` readback latency
- Per-dispatch submission overhead
- Buffer allocation round-trips

At large compute sizes the bandwidth/latency is amortized; at small sizes it dominates.

---

## 2. Results

### 2.1 Multi-Size Scaling: `np.matmul` (Deno WebGPU)

Naive matmul (Mul→Reduce kernel fusion, no shared memory tiling):

| N    | ms/call | GFLOP/s | Notes                          |
| ---- | ------- | ------- | ------------------------------ |
| 64   | 0.76    | 0.65    | Dispatch-bound (TB3 latency)   |
| 128  | 0.77    | 5.5     | Still dispatch-bound           |
| 256  | 0.81    | 41      | Transitioning to compute-bound |
| 512  | 1.23    | 219     | Compute-dominant               |
| 1024 | 4.76    | 452     | Good utilization               |

Peak theoretical: ~40 TFLOP/s (RTX 4070 Ti SUPER f32). Achieved **~1.1% of peak at 1024×1024** —
expected for naive row×col without shared memory tiling.

### 2.2 Tiled Matmul Variants (Deno WebGPU)

`lax.tiledMatmul` via `block_map` fused shader, with and without register tiling (`threadTile`):

| N    | np.matmul    | tiled(16) 16×16 | tt=[4,4] 64×64 | Best vs Naive      |
| ---- | ------------ | --------------- | -------------- | ------------------ |
| 256  | 39.7 GFLOP/s | **44.8**        | 28.2           | tiled 1.13×        |
| 512  | **222.5**    | 218.6           | 152.6          | naive wins         |
| 1024 | 451.6        | 519.5           | **564.0**      | **tt=[4,4] 1.25×** |
| 2048 | 655.2        | **758.3**       | 715.5          | **tiled 1.16×**    |

**Key observations:**

- Tiling helps at N ≥ 1024. Below that, kernel launch overhead dominates the benefit.
- `tt=[4,4]` (register tiling) peaks at 1024 (564 GFLOP/s = **1.25× naive**).
- At 2048, simple `tiled(16)` beats `tt=[4,4]` — register tiling overhead (fori_loop barrier +
  shared memory loads per iteration) hurts when tile count is large. The 64×64 tiles produce only
  1024 workgroups while 16×16 tiles produce 16384, enabling better GPU occupancy/overlap.
- **None reach the P1 target** (40% of peak = ~16 TFLOP/s). Next optimizations needed: vec4 loads,
  bank padding, larger thread tiles, and workgroup count tuning.

### 2.3 JS Dispatch Overhead (Deno WebGPU)

| Metric                                | Value        |
| ------------------------------------- | ------------ |
| Single call JS dispatch (warm)        | 0.17–0.45 ms |
| Single call GPU readback (`mapAsync`) | 12 ms        |
| 20 calls batched JS dispatch          | 0.18 ms/call |
| 20 calls total (dispatch + wait)      | 1.09 ms/call |

**JS overhead is negligible.** The 0.18 ms/call dispatch cost is <15% of the 1.23 ms total for
512×512. The Executable caching added in this session avoids redundant WGSL codegen.

### 2.4 Chromium Performance (Playwright, raw WebGPU shaders)

Both headed and headless Chromium achieve full hardware GPU performance when tested with raw WebGPU
dispatches served from a localhost page (secure context required):

| Config                        | 256²        | 512²        | 1024²       | Dispatch @200 |
| ----------------------------- | ----------- | ----------- | ----------- | ------------- |
| **Headed**                    | 275 GFLOP/s | 639 GFLOP/s | 773 GFLOP/s | 0.055 ms      |
| **Headless NEW**              | 305 GFLOP/s | 663 GFLOP/s | 781 GFLOP/s | 0.051 ms      |
| **Headless + no gpu-sandbox** | 305 GFLOP/s | 639 GFLOP/s | 784 GFLOP/s | 0.050 ms      |

**Key insight:** An earlier session observed headed Chromium at 400–1300 ms/call through vitest.
That was the jax-js JIT pipeline overhead (tracing → Jaxpr → codegen → compile → dispatch), not
Chrome's GPU performance. Raw dispatch latency is **0.05 ms** — comparable to Deno.

### 2.5 Chromium Secure Context Requirement

**Critical finding:** `navigator.gpu` is `undefined` on `about:blank` in Chromium 136+.

| Page                 | `isSecureContext` | `navigator.gpu` |
| -------------------- | ----------------- | --------------- |
| `about:blank`        | `false`           | `undefined`     |
| `http://localhost:*` | `true`            | `object` ✅     |
| `https://*`          | `true`            | `object` ✅     |

WebGPU requires a secure context. Vitest's dev server (localhost) provides this automatically.
Direct Playwright probes navigating to `about:blank` will falsely report no WebGPU support.

---

## 3. Headless Chromium WebGPU: Configuration

### Chrome Flags for Headless GPU

Per
[developer.chrome.com/blog/supercharge-web-ai-testing](https://developer.chrome.com/blog/supercharge-web-ai-testing):

```js
args: [
  "--no-sandbox",
  "--headless=new", // New headless mode (not legacy)
  "--use-angle=vulkan", // Vulkan backend for ANGLE
  "--enable-features=Vulkan", // Enable Vulkan compositing
  "--disable-vulkan-surface", // Skip VK_KHR_surface/swapchain
  "--enable-unsafe-webgpu", // Enable WebGPU
];
```

### Status: Implemented ✅

`vitest.config.ts` updated to headless mode with full flags. Verified:

- NVIDIA Lovelace adapter detected (not SwiftShader)
- `subgroups` feature available
- All 42 core tests pass (basic + multi-kernel)
- Test startup 5× faster (~1.1s vs ~5.8s headed)
- GPU performance identical to headed mode

---

## 4. Action Items

| Priority | Item                                                                | Status                          |
| -------- | ------------------------------------------------------------------- | ------------------------------- |
| ✅       | Deno WebGPU for GPU performance benchmarks                          | Done                            |
| ✅       | Headless Chromium with full flag set                                | Done — identical perf to headed |
| ✅       | Switch vitest to headless mode                                      | Done — 5× faster startup        |
| **Low**  | Investigate `timestamp-query` for GPU-side timing                   | Future                          |
| **Note** | Deno lacks `subgroups` — subgroup optimizations need Chromium tests | Documented                      |

---

## 5. Deno vs Chromium Feature Comparison

| Feature                           | Deno (wgpu-rs) | Chromium (Dawn)          |
| --------------------------------- | -------------- | ------------------------ |
| `subgroups`                       | ❌             | ✅                       |
| `shader-f16`                      | ✅             | ✅                       |
| `timestamp-query`                 | ✅             | ✅                       |
| `dual-source-blending`            | ✅             | ✅                       |
| `float32-filterable`              | ✅             | ✅                       |
| `maxStorageBuffersPerShaderStage` | 1048576        | 10                       |
| Headless GPU support              | ✅ (native)    | ✅ (with full flags)     |
| Benchmark reliability             | ✅ Consistent  | ✅ Consistent (headless) |

The `maxStorageBuffers` difference is significant: Deno's wgpu-rs reports the Vulkan hardware limit
(~1M) while Chromium caps at the WebGPU spec minimum (10). This means `splitGraphDataflow` P2 will
rarely trigger on Deno but may split fused kernels on Chromium.
