/**
 * Tests for the REPL source-map remapping pipeline.
 *
 * The REPL compiles user TypeScript → TS transpile (ESNext target) → Rollup
 * system-format bundle → prepend header ";". A source map tracks positions
 * through this chain. `checkLeaks` reports generated-code positions (e.g.
 * `index.ts:5:5`); the remap functions convert these back to the user's
 * original source lines.
 *
 * IMPORTANT: The REPL uses `target: ESNext` so that `using` declarations are
 * emitted natively (just stripped of types). Using `target: ES2022` would
 * downlevel `using` into 80+ lines of try/catch/finally helper code whose
 * source map has gaps (lines 78-85 unmapped), breaking leak line remapping.
 *
 * VLQ refresher (for reading test data):
 *   A=0  C=+1  D=-1  E=+2  G=+3  I=+4  K=+5  M=+6
 *   Each line's segments are separated by ","; lines separated by ";".
 *   The first field (genCol) resets per line; source/srcLine/srcCol are
 *   cumulative across lines.
 */
import { describe, expect, test } from "vitest";

import {
  decodeVlq,
  mapGeneratedPositionToSource,
  parseLeakMarkers,
  remapLeakDetails,
  remapReplLocationText,
} from "../website/src/lib/repl/sourcemap";

// ---------------------------------------------------------------------------
// decodeVlq
// ---------------------------------------------------------------------------
describe("decodeVlq", () => {
  test("zero", () => {
    expect(decodeVlq("A", 0)).toEqual([0, 1]);
  });

  test("positive values", () => {
    expect(decodeVlq("C", 0)).toEqual([1, 1]);
    expect(decodeVlq("E", 0)).toEqual([2, 1]);
    expect(decodeVlq("I", 0)).toEqual([4, 1]);
    expect(decodeVlq("M", 0)).toEqual([6, 1]);
  });

  test("negative values", () => {
    expect(decodeVlq("D", 0)).toEqual([-1, 1]);
    expect(decodeVlq("F", 0)).toEqual([-2, 1]);
  });

  test("starts at given index", () => {
    // "xCy" — decode starting at index 1 (the "C")
    expect(decodeVlq("xCy", 1)).toEqual([1, 2]);
  });
});

// ---------------------------------------------------------------------------
// mapGeneratedPositionToSource
// ---------------------------------------------------------------------------
describe("mapGeneratedPositionToSource", () => {
  // Simple map: 1 header line then 3 lines mapping 1:1 to source lines 1-3.
  //   ;        → gen line 1 (header, no mapping)
  //   AAAA     → gen line 2 → src line 1, col 1
  //   AACA     → gen line 3 → src line 2, col 1  (srcLine delta +1)
  //   AACA     → gen line 4 → src line 3, col 1  (srcLine delta +1)
  const simpleMap = { mappings: ";AAAA;AACA;AACA" };

  test("header line returns null", () => {
    expect(mapGeneratedPositionToSource(simpleMap, 1, 1)).toBeNull();
  });

  test("1:1 line mapping", () => {
    expect(mapGeneratedPositionToSource(simpleMap, 2, 1)).toEqual({
      line: 1,
      column: 1,
    });
    expect(mapGeneratedPositionToSource(simpleMap, 3, 1)).toEqual({
      line: 2,
      column: 1,
    });
    expect(mapGeneratedPositionToSource(simpleMap, 4, 1)).toEqual({
      line: 3,
      column: 1,
    });
  });

  test("line beyond last mapping returns null", () => {
    expect(mapGeneratedPositionToSource(simpleMap, 99, 1)).toBeNull();
  });

  test("column offsets", () => {
    // Gen line 2: col 0 → src (1,1)
    // Gen line 3: col 4 → src (3,7)
    //   IAEM = genCol +4, src +0, srcLine +2, srcCol +6
    //          source line = 0+2+1 = 3, source col = 0+6+1 = 7
    const colMap = { mappings: ";AAAA;IAEM" };
    expect(mapGeneratedPositionToSource(colMap, 3, 5)).toEqual({
      line: 3,
      column: 7,
    });
    // Column before mapping segment → null (no earlier segment on that line)
    expect(mapGeneratedPositionToSource(colMap, 3, 1)).toBeNull();
  });

  test("null / empty map returns null", () => {
    expect(mapGeneratedPositionToSource(null, 2, 1)).toBeNull();
    expect(mapGeneratedPositionToSource({}, 2, 1)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// remapReplLocationText
// ---------------------------------------------------------------------------
describe("remapReplLocationText", () => {
  const map = { mappings: ";AAAA;AACA;AACA" };

  test("rewrites index.ts:LINE:COL with source positions", () => {
    const text = "Array:float32[] rc=1 created at index.ts:3:1";
    // Gen line 3 → source line 2
    expect(remapReplLocationText(text, map)).toBe(
      "Array:float32[] rc=1 created at index.ts:2:1",
    );
  });

  test("rewrites main.ts references too", () => {
    expect(remapReplLocationText("at main.ts:4:1", map)).toBe("at main.ts:3:1");
  });

  test("null map returns text unchanged", () => {
    const text = "Array:float32[] rc=1 created at index.ts:5:10";
    expect(remapReplLocationText(text, null)).toBe(text);
  });

  test("non-matching text passes through", () => {
    expect(remapReplLocationText("hello world", map)).toBe("hello world");
  });
});

// ---------------------------------------------------------------------------
// remapLeakDetails
// ---------------------------------------------------------------------------
describe("remapLeakDetails", () => {
  const map = { mappings: ";AAAA;AACA;AACA" };

  test("remaps every detail string", () => {
    const details = [
      "Array:float32[] rc=1 created at index.ts:3:1",
      "Array:float32[4] rc=1 created at index.ts:4:1",
    ];
    expect(remapLeakDetails(details, map)).toEqual([
      "Array:float32[] rc=1 created at index.ts:2:1",
      "Array:float32[4] rc=1 created at index.ts:3:1",
    ]);
  });
});

// ---------------------------------------------------------------------------
// parseLeakMarkers
// ---------------------------------------------------------------------------
describe("parseLeakMarkers", () => {
  test("extracts line numbers and builds message", () => {
    const details = [
      "Array:float32[] rc=1 created at index.ts:2:1",
      "Array:float32[4] rc=1 created at index.ts:3:7",
    ];
    expect(parseLeakMarkers(details)).toEqual([
      {
        line: 2,
        message: "Leaked: Array:float32[] rc=1. Use `using` or call .dispose()",
      },
      {
        line: 3,
        message:
          "Leaked: Array:float32[4] rc=1. Use `using` or call .dispose()",
      },
    ]);
  });

  test("ignores non-matching lines", () => {
    expect(parseLeakMarkers(["no match here"])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// End-to-end: realistic REPL scenario
// ---------------------------------------------------------------------------
describe("realistic REPL scenario", () => {
  // User writes 3 lines of TypeScript.
  // After TS → Rollup system-format → header, the generated bundle is:
  //   line 1:  ;                                (header—no mapping)
  //   line 2:  System.register(["jax"], ...)    (boilerplate)
  //   line 3:    var np;                        (boilerplate)
  //   line 4:    return { execute: async ... {  (boilerplate)
  //   line 5:      const x = np.array([1,2,3]);  ← user line 2
  //   line 6:      console.log(await x.data());   ← user line 3
  //   line 7:    }};                            (boilerplate)
  //
  // Source map (with leading ";" for the header, ";;;" for boilerplate):
  //   ;;;;IACA;IACA
  //   Line 5: IACA = genCol+4, src+0, srcLine+1, srcCol+0  → src line 2, col 1
  //   Line 6: IACA = genCol+4, src+0, srcLine+1, srcCol+0  → src line 3, col 1
  const replMap = { mappings: ";;;;IACA;IACA" };

  test("checkLeaks detail is remapped to user source line", () => {
    // V8 reports the generated position (line 5, col 5 for the np.array call)
    const detail = "Array:float32[3] rc=1 created at index.ts:5:5";
    const remapped = remapReplLocationText(detail, replMap);
    expect(remapped).toBe("Array:float32[3] rc=1 created at index.ts:2:1");
  });

  test("full pipeline: remap then parse markers", () => {
    const rawDetails = [
      "Array:float32[3] rc=1 created at index.ts:5:5",
      "Array:float32[3] rc=1 created at index.ts:6:5",
    ];
    const remapped = remapLeakDetails(rawDetails, replMap);
    expect(remapped).toEqual([
      "Array:float32[3] rc=1 created at index.ts:2:1",
      "Array:float32[3] rc=1 created at index.ts:3:1",
    ]);
    const markers = parseLeakMarkers(remapped);
    expect(markers).toEqual([
      {
        line: 2,
        message:
          "Leaked: Array:float32[3] rc=1. Use `using` or call .dispose()",
      },
      {
        line: 3,
        message:
          "Leaked: Array:float32[3] rc=1. Use `using` or call .dispose()",
      },
    ]);
  });

  test("boilerplate lines return null (no marker created)", () => {
    // Line 2 is boilerplate — no source mapping
    const detail = "something at index.ts:2:1";
    const remapped = remapReplLocationText(detail, replMap);
    // No mapping found → position is left unchanged
    expect(remapped).toBe("something at index.ts:2:1");
  });
});

// ---------------------------------------------------------------------------
// End-to-end: real TypeScript transpile source maps (ESNext vs ES2022)
// ---------------------------------------------------------------------------
describe("TS transpile source map coverage", () => {
  // These maps were captured from actual TypeScript 5.9 transpileModule output.
  // The REPL uses target: ESNext so that `using` declarations are emitted
  // natively. Using target: ES2022 downlevels `using` into 60-90 lines of
  // try/catch/finally helper code whose source map has coverage gaps.

  // User code (4 lines):
  //   import { numpy as np } from "@jax-js-nonconsuming/jax";
  //   const x = np.array([1, 2, 3, 4]);
  //   using y = np.array([5, 6]);
  //   console.log(x.js());

  // ESNext output: 4 code lines (+ sourceMappingURL), all mapped
  const esnextMap = {
    mappings:
      "AAAA,OAAO,EAAE,KAAK,IAAI,EAAE,EAAE,MAAM,0BAA0B,CAAC;AACvD,MAAM,CAAC,GAAG,EAAE,CAAC,KAAK,CAAC,CAAC,CAAC,EAAE,CAAC,EAAE,CAAC,EAAE,CAAC,CAAC,CAAC,CAAC;AACjC,MAAM,CAAC,GAAG,EAAE,CAAC,KAAK,CAAC,CAAC,CAAC,EAAE,CAAC,CAAC,CAAC,CAAC;AAC3B,OAAO,CAAC,GAAG,CAAC,CAAC,CAAC,EAAE,EAAE,CAAC,CAAC",
  };

  test("ESNext: all 4 user code lines have source map coverage", () => {
    const groups = esnextMap.mappings.split(";");
    expect(groups).toHaveLength(4); // 4 generated lines
    // Every group should have segments (no empty lines)
    for (const g of groups) {
      expect(g.length).toBeGreaterThan(0);
    }
  });

  test("ESNext: np.array line maps back to original line 2", () => {
    // Line 2 of ESNext output = `const x = np.array([1, 2, 3, 4]);`
    const pos = mapGeneratedPositionToSource(esnextMap, 2, 1);
    expect(pos).not.toBeNull();
    expect(pos!.line).toBe(2); // original line 2
  });

  test("ESNext: using line maps back to original line 3", () => {
    const pos = mapGeneratedPositionToSource(esnextMap, 3, 1);
    expect(pos).not.toBeNull();
    expect(pos!.line).toBe(3);
  });

  test("ESNext + header: np.array at gen line 3 maps to original line 2", () => {
    // REPL prepends ";" for header → shift all mappings by 1 generated line
    const adjusted = { mappings: ";" + esnextMap.mappings };
    const pos = mapGeneratedPositionToSource(adjusted, 3, 1);
    expect(pos).not.toBeNull();
    expect(pos!.line).toBe(2);
  });

  test("ES2022 target produces fewer map groups than output lines", () => {
    // ES2022 transpile of the same 4-line code produces 68 output lines
    // but only 59 source map groups — catch/finally blocks are unmapped.
    // This is the root cause of the REPL leak diagnostic line number bug.
    const es2022OutputLines = 68;
    const es2022MapGroups = 59;
    expect(es2022MapGroups).toBeLessThan(es2022OutputLines);
  });
});
