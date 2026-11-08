import "vitest";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

// Vitest 5 form: augmenting `Matchers` adds the matchers to `expect(x)`,
// asymmetric matchers and `expect.extend` at once. `skipLibCheck` hides
// errors in this file, so check it with `tsc --skipLibCheck false` after edits.
declare module "vitest" {
  // Merged declarations must repeat vitest's type parameter names; renaming
  // the unused `T` to `_T` stops the merge and the matchers disappear.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface Matchers<R, T> {
    toBeAllclose: (
      expected: Parameters<typeof np.array>[0],
      options?: { rtol?: number; atol?: number; equalNaN?: boolean },
    ) => R;
    toBeWithinRange: (min: number, max: number) => R;
  }
}
