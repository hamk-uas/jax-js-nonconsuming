/**
 * @file Symbolic dimension types for polymorphic shapes (M4.1).
 *
 * This module is dependency-free (no imports from other src/ modules) to avoid
 * circular imports — both shape.ts and utils.ts depend on it.
 */

/**
 * A symbolic dimension variable used for polymorphic shapes.
 * During tracing with `jit({ dynamic_axes })`, dynamic axes are
 * represented as SymDim instances instead of concrete numbers.
 * SymDim values are resolved to concrete numbers before compilation.
 */
export class SymDim {
  constructor(readonly name: string) {}
  toString(): string {
    return this.name;
  }
  toJSON(): string {
    return `__sym__${this.name}`;
  }
}

/**
 * A dimension is either a concrete number or a symbolic variable.
 * ShapedArray.shape uses Dim[] to support polymorphic shapes.
 */
export type Dim = number | SymDim;

/** Check if a dimension is symbolic. */
export function isSymbolicDim(d: Dim): d is SymDim {
  return d instanceof SymDim;
}

/** Returns true if any dimension in the shape is symbolic. */
export function hasSymbolicDims(shape: readonly Dim[]): boolean {
  return shape.some(isSymbolicDim);
}

/**
 * Assert that a dimension is concrete, returning the number.
 * Throws if the dimension is symbolic.
 */
export function concreteDim(d: Dim, context?: string): number {
  if (typeof d === "number") return d;
  throw new Error(
    `Expected concrete dimension but got symbolic dim '${d.name}'${context ? ` in ${context}` : ""}`,
  );
}

/**
 * Assert that all dimensions in a shape are concrete.
 * Returns the shape typed as number[]. Throws if any dimension is symbolic.
 */
export function concreteShape(
  shape: readonly Dim[],
  context?: string,
): number[] {
  for (const d of shape) {
    if (typeof d !== "number") {
      throw new Error(
        `Expected concrete shape but got symbolic dim '${(d as SymDim).name}' in [${shape}]${context ? ` (${context})` : ""}`,
      );
    }
  }
  return shape as number[];
}

/**
 * Compare two dimensions for equality. Two SymDims are equal if they have
 * the same name.
 */
export function dimEquals(a: Dim, b: Dim): boolean {
  if (typeof a === "number" && typeof b === "number") return a === b;
  if (a instanceof SymDim && b instanceof SymDim) return a.name === b.name;
  return false;
}

/**
 * Resolve a shape by substituting symbolic dimensions with concrete values.
 */
export function resolveShape(
  shape: readonly Dim[],
  bindings: ReadonlyMap<string, number>,
): number[] {
  return shape.map((d) => {
    if (typeof d === "number") return d;
    const val = bindings.get(d.name);
    if (val === undefined) {
      throw new Error(
        `Unresolved symbolic dimension '${d.name}' — not in bindings`,
      );
    }
    return val;
  });
}
