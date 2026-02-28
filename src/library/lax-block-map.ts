// lax.blockMap — apply a function independently to each block of inputs.
//
// Partitions inputs along grid axes into blocks of `blockShape`, applies
// the body to each block, and reassembles the results.

import { Array } from "../frontend/array";
import { bind, getAval, Primitive, ShapedArray } from "../frontend/core";
import { makeJaxpr } from "../frontend/jaxpr";
import * as tree from "../tree";
import type { JsTree } from "../tree";

const JsArray = globalThis.Array;

/**
 * Options for {@link blockMap}.
 */
export interface BlockMapOptions {
  /**
   * The block size along each grid axis. `blockShape[g]` is the number of
   * elements processed per block along grid axis `g`.
   */
  blockShape: number[];

  /**
   * Per-input mapping from grid axes to input dimensions.
   *
   * - A single `(number | null)[]` is broadcast to all input leaves.
   * - An array of `(number | null)[][]` provides per-leaf mappings (length
   *   must match the number of flattened input leaves).
   *
   * For each leaf, `inAxes[g]` is the dimension mapped to grid axis `g`,
   * or `null` if that grid axis doesn't slice this input (broadcast).
   *
   * @default `[0, 1, ..., gridRank-1]` for all leaves.
   */
  inAxes?: (number | null)[] | (number | null)[][];

  /**
   * Per-output mapping from grid axes to output dimensions. Same format as
   * `inAxes` but applied to body outputs.
   *
   * @default Same as `inAxes` default.
   */
  outAxes?: (number | null)[] | (number | null)[][];
}

/**
 * Apply `f` independently to blocks of `elems`, tiled along a grid defined
 * by `blockShape` and `inAxes`/`outAxes`.
 *
 * @example Elementwise doubling in blocks of 4
 * ```ts
 * const result = lax.blockMap(
 *   (block) => block.mul(np.array(2)),
 *   xs,
 *   { blockShape: [4] },
 * );
 * ```
 */
export function blockMap<I extends JsTree<Array>, O extends JsTree<Array>>(
  f: (block: I) => O,
  elems: I,
  options: BlockMapOptions,
): O {
  const { blockShape } = options;
  const gridRank = blockShape.length;

  // Flatten elems pytree
  const [flatElems, elemsTreedef] = tree.flatten<Array>(elems);
  const numInputs = flatElems.length;
  if (numInputs === 0) {
    throw new Error("blockMap: elems must have at least one leaf array");
  }

  // Resolve inAxes
  const defaultAxes: (number | null)[] = JsArray.from(
    { length: gridRank },
    (_, i) => i,
  );
  const flatInAxes = resolveAxes(
    options.inAxes,
    numInputs,
    defaultAxes,
    "inAxes",
  );

  // Compute block-shaped abstract values for tracing
  const blockAvals: ShapedArray[] = flatElems.map((elem, i) => {
    const aval = getAval(elem);
    const shape = [...(aval.shape as number[])];
    for (let g = 0; g < gridRank; g++) {
      if (flatInAxes[i][g] !== null) {
        shape[flatInAxes[i][g]!] = blockShape[g];
      }
    }
    return new ShapedArray(shape, aval.dtype, aval.weakType);
  });

  // Trace the body function
  let outTreedef: tree.JsTreeDef | undefined;
  const traceFn = (...blockFlat: Array[]): Array[] => {
    const blockPytree = tree.unflatten(elemsTreedef, blockFlat) as I;
    const out = f(blockPytree);
    const [outFlat, td] = tree.flatten<Array>(out as JsTree<Array>);
    outTreedef = td;
    return outFlat;
  };

  const { jaxpr: closedJaxpr } = makeJaxpr(traceFn)(...blockAvals);
  const jaxpr = closedJaxpr.jaxpr;
  const consts = closedJaxpr.consts;
  const numConsts = consts.length;

  // Resolve outAxes
  const numOutputs = jaxpr.outs.length;
  const flatOutAxes = resolveAxes(
    options.outAxes,
    numOutputs,
    defaultAxes,
    "outAxes",
  );

  // Call the BlockMap primitive
  const results = bind(Primitive.BlockMap, [...consts, ...flatElems], {
    jaxpr,
    blockShape,
    inAxes: flatInAxes,
    outAxes: flatOutAxes,
    numConsts,
    numInputs,
  }) as Array[];

  // Dispose the captured consts from tracing
  closedJaxpr.dispose();

  // Reconstruct output pytree
  return tree.unflatten(outTreedef!, results) as O;
}

/** Resolve a user-provided axes spec into a flat (number|null)[][] array. */
function resolveAxes(
  axes: (number | null)[] | (number | null)[][] | undefined,
  count: number,
  defaultAxes: (number | null)[],
  name: string,
): (number | null)[][] {
  if (axes === undefined) {
    return JsArray.from({ length: count }, () => [...defaultAxes]);
  }
  // Single axes spec (1D array of number|null) → broadcast to all leaves
  if (axes.length > 0 && !JsArray.isArray(axes[0])) {
    const single = axes as (number | null)[];
    return JsArray.from({ length: count }, () => [...single]);
  }
  // Array of per-leaf specs
  const multi = axes as (number | null)[][];
  if (multi.length !== count) {
    throw new Error(
      `blockMap: ${name} has ${multi.length} entries but expected ${count}`,
    );
  }
  return multi;
}
