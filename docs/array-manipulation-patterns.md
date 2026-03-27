# Array manipulation patterns

This document covers structural array reads and writes - the right way to extract rows, columns, and
slices, and the right way to update them. It exists because the most natural-looking alternatives
(one-hot matrix algebra, repeated piecemeal patching, or overusing restack workarounds) are easy to
reach for in JavaScript/TypeScript, but not all of them have the same cost profile.

One important distinction up front:

- Building a small fixed matrix from known scalars or vectors with `np.array([...])` or
  `np.stack([...])` is usually fine.
- Using dense algebra to fake indexing or mutation is not.
- Repeated dynamic patching is where the current public API still has real gaps versus JAX's `.at[]`.

## The anti-pattern: one-hot algebra

When JavaScript/TypeScript lacks `A[:, j]` syntax, it is tempting to simulate extraction and
mutation with dense algebra:

```ts
// Extract column j by multiplying by a one-hot column vector:
using col_j = np.matmul(A, oneHotCol[j]); // O(n²) matmul of mostly zeros

// "Set" column j by adding a rank-1 outer product:
using patch = np.matmul(newCol, oneHotRow[j]); // O(n²) matmul
using A_new = np.add(A, patch); // O(n²) add
```

This is algebraically correct but operationally wrong:

1. **Algorithmic cost.** Extracting column `j` is an O(n) memory read. Writing column `j` is an
   O(n) memory write. The one-hot approach turns both into O(n²) dense matrix operations over
   mostly-zero data.

2. **Fusion barriers.** `np.matmul` is not elementwise, so the JIT compiler cannot fuse it into
   surrounding elementwise chains. Each one-hot matmul forces a separate dispatch, serializing
   memory buffers and blocking kernel fusion.

3. **Memory churn.** Synthesizing full [n, n] intermediate matrices wastes both VRAM bandwidth and
   allocation budget on data that is structurally zero.

This is the pattern to avoid in downstream code. By contrast, `diffSVMC`'s current 5x5 Yasso matrix
assembly via five row stacks is not the same anti-pattern: it is a small fixed construction from known
scalars, not an O(n^2) surrogate for a row/column read or write.

## Structural reads

Use slicing and indexing primitives instead of algebraic extraction.

### Single row or column

```ts
// Row i (removes the indexed axis, returns shape [n]):
using row_i = lax.dynamicIndexInDim(A, i, 0);

// Column j (removes the indexed axis, returns shape [m]):
using col_j = lax.dynamicIndexInDim(A, j, 1);

// Keep the axis as size-1 (returns shape [1, n] or [m, 1]):
using row_i = lax.dynamicIndexInDim(A, i, 0, true);
using col_j = lax.dynamicIndexInDim(A, j, 1, true);
```

`lax.dynamicIndexInDim` is a zero-copy view (Shrink + reshape). O(1) in eager mode, fuses in JIT.

### Contiguous slice

```ts
// Rows 2..5 along axis 0:
using block = lax.sliceInDim(A, 2, 5, 0);

// Columns 0..3 along axis 1:
using block = lax.sliceInDim(A, 0, 3, 1);
```

`lax.sliceInDim` is also a zero-copy view. Negative indices and omitted limits are supported.

### Arbitrary-start multi-axis slice (runtime indices)

```ts
// Extract a [2, 3] block starting at runtime row r, column c:
using block = lax.dynamicSlice(A, [r, c], [2, 3]);
```

`lax.dynamicSlice` takes scalar Array start indices (one per axis) and concrete slice sizes. Use
this when start positions are runtime values (e.g. from a traced loop).

### Gather by index array

```ts
// Gather rows at arbitrary integer indices:
using selected = np.take(A, indices, 0);
```

`np.take` is the right tool when indexing by an integer array rather than a contiguous range.

## Structural writes

Use `lax.dynamicUpdateSlice` instead of one-hot expansion + addition.

### Single row or column update

```ts
// Set column j of A to newCol (shape [m] or [m, 1]):
using col2d = np.reshape(newCol, [m, 1]); // ensure [m, 1] shape
using A_new = lax.dynamicUpdateSlice(A, col2d, j, 1);

// Set row i of A to newRow (shape [n] or [1, n]):
using row2d = np.reshape(newRow, [1, n]); // ensure [1, n] shape
using A_new = lax.dynamicUpdateSlice(A, row2d, i, 0);
```

`lax.dynamicUpdateSlice` patches a contiguous slice along one axis. Under JIT, this lowers to a
zero-copy DUS step (axis 0) or a fiber-loop copy (axis > 0). In eager mode, it copies the full
array with the slice overwritten.

### Replacing a contiguous block

```ts
// Replace rows 2..4 with a [2, n] patch:
using A_new = lax.dynamicUpdateSlice(A, patch, 2, 0);
```

The `src` shape must match `dst` on all axes except the update axis, where it determines the
slice width.

### ND block update (multi-axis)

```ts
// Replace a [2, 3] interior block starting at row 1, column 2:
using A_new = lax.dynamicUpdateSlice(A, block, [1, 2]);
```

The ND form accepts a `number[]` of per-axis start indices. `src` must have the same rank as
`dst`, and `src.shape[i]` determines the slice width along each axis. This is the equivalent of
JAX's `A.at[1:3, 2:5].set(block)` for static offsets.

Under JIT, multi-axis updates lower to a single ND-aware DUS step with precomputed stride
decomposition. Axes where `src` spans the full `dst` extent are optimized away (no patching
needed on those axes). The single-axis fast path (zero-copy for axis 0, fiber loop for axis > 0)
activates automatically when only one axis is active.

Fully differentiable: JVP, `grad`, and `vmap` are all supported for the ND form.

### Current limitations

- The `offset` / `startIndices` parameters are **plain JS numbers**. This covers the verified
  downstream use cases (plain `for` loops, fixed assembly), but does not support traced Array
  offsets from `lax.foriLoop`. If you need a runtime offset from a traced loop, accumulate
  results and stack.
- Eager-mode update copies the full destination array. The O(n) cost advantage is strongest under
  JIT.

## What JAX `.at[]` still provides that we do not

JAX's `.at[...]` interface is broader than the current public update surface here.

The gaps below are ordered by verified downstream impact, not by abstract JAX parity:

1. ~~**ND block updates.**~~ Resolved. `lax.dynamicUpdateSlice(dst, src, [s0, s1, ...])`
   supports multi-axis block updates with static offsets, full AD, and JIT optimization.
2. **Traced write offsets (future parity).** `lax.dynamicSlice` accepts traced scalar start
   indices for reads, but `lax.dynamicUpdateSlice` requires plain JS numbers for writes. This
   is a real API asymmetry versus JAX, but it has not been demonstrated as a blocker by current
   downstream code (`diffSVMC`, `dlm-js`). Their loop indices are plain JS numbers, not traced
   values.
3. **General scatter-set style updates (future).** We expose `scatterAdd`, but not the broader
   family of scatter-set/scatter-min/scatter-max. Evaluate demand as needed.

These gaps matter most for multi-axis piecemeal construction. For small fixed analytical
constructions, `np.array([...])` and `np.stack([...])` remain the right tools.

## Reference semantics and future API design

This repo uses a non-consuming public ownership model: update helpers return new arrays and do not
consume or invalidate the original input.

That should remain true even if we add a closer bridge to JAX's `.at[]`.

- A future update helper may lower to an in-place mutate/recycle path under `jit()`.
- But the user-visible semantics should stay pure and non-consuming.
- We should not add an in-place mutation API just to mimic Python indexing syntax.

That design constraint affects prioritization. The right sequence is:

1. ~~add ND support to structural update primitives (static/JS-number offsets first)~~ Done.
2. extend to traced offsets only when downstream evidence justifies it
3. consider ergonomic builder sugar last, judged on clarity rather than literal JAX syntax parity

## Small matrix assembly

For building small fixed-size matrices from known scalars or vectors, use `np.stack` or
`np.array`:

```ts
// 2×2 from scalars:
using M = np.array([[a, b], [c, d]]);

// Stack row vectors into a matrix:
using M = np.stack([row0, row1, row2]);

// Stack column vectors:
using M = np.stack([col0, col1, col2], 1);
```

This is the right pattern for small analytical constructions (Jacobians, rotation matrices,
Cramer's rule). The library uses it internally for analytical inverse, Cholesky, and QR at small
sizes.

This also covers small downstream constructions such as `diffSVMC`'s current Yasso 5x5 matrix
assembly from known scalar entries. That code may still benefit from better update ergonomics in the
future, but it is not performance-equivalent to one-hot algebra and does not need to be rewritten
just for style.

## Accumulate-then-stack (loop output assembly)

When building a matrix row-by-row or column-by-column inside a loop, prefer collecting results and
stacking once over repeated in-place updates:

```ts
// Good: collect then stack
const rows: Array[] = [];
for (let i = 0; i < n; i++) {
  rows.push(computeRow(i));
}
using M = np.stack(rows);
rows.forEach((r) => r.dispose());

// Also good: use lax.scan to produce stacked outputs directly
const [_, rows] = await lax.scan(
  (carry, x) => [carry, computeRow(carry, x)],
  init,
  xs,
);
```

Under `lax.scan`, the stacked Y outputs are written directly into a pre-allocated buffer — no
intermediate concatenation needed.

## Summary

| Operation              | Anti-pattern                   | Correct primitive                 | Cost    |
| ---------------------- | ------------------------------ | --------------------------------- | ------- |
| Extract row/column     | `matmul(A, oneHot)`            | `lax.dynamicIndexInDim(A, i, ax)` | O(n)    |
| Extract contiguous     | `matmul` + masking             | `lax.sliceInDim(A, lo, hi, ax)`   | O(1)    |
| Extract by indices     | `matmul` + selector matrix     | `np.take(A, idx, ax)`             | O(k)    |
| Update row/column      | `add(A, matmul(v, oneHotRow))` | `lax.dynamicUpdateSlice(A,v,i,ax)`| O(n)    |
| Assemble small matrix  | per-element `set` loop         | `np.array([...])` / `np.stack`    | O(n²)   |
| Loop output collection | repeated update-slice          | `lax.scan` with Y outputs         | O(n)    |

## Practical rule of thumb

- If the structure is small and statically known, `np.array([...])` or `np.stack([...])` is fine.
- If you are expressing a read or write through zeros, one-hot vectors, or dense algebra, stop and
  use structural primitives instead.
- If the natural expression needs runtime ND updates and the public API feels awkward, that is a
  real library gap, not necessarily a downstream code smell.
