# Remaining Work Plan

Consolidated from 6 completed plan files. All major architectural work is done. What remains is
performance optimization opportunities that depend on browser spec maturity.

## P7: Subgroup Matrix Ops (WMMA)

**Priority:** Low — blocked on Chrome 144+ spec stability **Source:**
COMPILING-EFFICIENT-SHADERS-PLAN Phase 6

Hardware tensor cores (`subgroupMatrixMultiply`) would give 2–4× improvement over the current tiled
matmul (which already hits 53.7% of peak FP32 at 4096×4096). The current implementation uses scalar
FMA in `var<private>` registers.

**Prerequisites:**

- `chromestatus.com/feature/subgroup-matrix` reaches "Enabled by default"
- Feature detection: `adapter.features.has("subgroups-matrix")`
- WGSL `enable subgroups_matrix;` available in stable Chrome

**Implementation sketch:**

1. Add `subgroups-matrix` to required features in device request
2. Replace inner K-tile loop in `emitTiledMatmulBody` with `subgroupMatrixMultiply`
3. Adjust `threadTile` and workgroup size for WMMA tile shapes (typically 16×16)
4. Feature-detect and fall back to current scalar FMA path

**Acceptance:** 4096×4096 f32 matmul at ≥70% of peak GFLOP/s on RTX 4070 Ti SUPER.

## P2: Relaxed SIMD FMA (WASM)

**Priority:** Medium — Safari doesn't support relaxed SIMD **Source:** copilot-instructions §WASM
feature opportunities

`f32x4.relaxed_madd` for 2× dot-product throughput in WASM matmul and scan bodies.

**Risk:** Safari is the only holdout. Acceptable if gated behind feature detection.

## P3: i64 in wasmblr

**Priority:** Medium — no browser risk (WASM MVP feature) **Source:** copilot-instructions §WASM
feature opportunities

Native i64 instructions in wasmblr. Unlocks f64 builtins, simplifies Threefry PRNG key mixing.

## P4: Conv2d Tuning

**Priority:** Medium **Source:** copilot-instructions §Future performance work

Tiled matmul already gives free improvement to im2col-based conv2d. Remaining work: specialized WGSL
shaders for common kernel sizes (3×3, 5×5) that avoid the im2col materialization.

## O3: Vec4 Vectorization (Deferred)

**Priority:** Low **Source:** COMPILING-EFFICIENT-SHADERS-PLAN

WGSL `vec4<f32>` loads/stores for 4× memory bandwidth. Deferred because current bottleneck is
compute (FMA throughput), not memory bandwidth, at the sizes where block_map is used.

## Completed Plans (Consolidated)

All of these are fully implemented and their plan files have been deleted:

| Plan File                                     | What It Did                                                       |
| --------------------------------------------- | ----------------------------------------------------------------- |
| `BLOCK-MAP-IMPLEMENTATION-PLAN-2.md`          | `block_map` primitive, eager + JIT + WebGPU fused codegen         |
| `COMPILING-EFFICIENT-SHADERS-PLAN.md`         | Shader optimization Phases 0–5, 53.7% peak FP32                   |
| `ASSOC-SCAN-UNIFORM-JAXPR-PLAN.md`            | Uniform jaxpr-based assocScan, legacy path deleted (−1761 LOC)    |
| `ASSOC-SCAN-DEBT-RETIREMENT-PLAN.md`          | Shared blocked-data-movement primitives + thin orchestrator       |
| `PLAN-1B-REINTRODUCTION-REVERSE-PRIMITIVE.md` | `Primitive.Reverse` for polymorphic-safe reverse                  |
| `SCAN-REDUCTIONS-PLAN.md`                     | Fused scan reductions via `var<private>` arrays + `createWgslGen` |
