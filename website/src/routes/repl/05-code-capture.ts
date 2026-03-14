import {
  type CodeCaptureEntry,
  jit,
  numpy as np,
  setCodeCapture,
} from "@hamk-uas/jax-js-nonconsuming";

// setCodeCapture() lets you inspect the generated Wasm or WebGPU shader code
// that the JIT compiler produces. Try switching between "WebGPU" and "Wasm"
// in the dropdown above to see WGSL vs WAT output!
//
// Note: The callback fires on compilation, not on every call. WebGPU shaders
// are cached (ShaderPipelineCache), so you'll only see output on the first run.
// Wasm mega-modules are compiled fresh each time, so they appear on every run.

const captured: CodeCaptureEntry[] = [];
setCodeCapture((entry) => captured.push(entry));

// A small JIT-compiled function — the compiler will generate native code for it.
using x = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
const f = jit((a: np.Array) => {
  using sq = a.mul(a);
  using shifted = sq.add(np.array([1, 1, 1, 1, 1, 1, 1, 1]));
  return np.sqrt(shifted);
});
using result = f(x);
f.dispose();

// Turn off capture when you're done.
setCodeCapture(null);

// Print what was captured.
for (const entry of captured) {
  console.log(`--- ${entry.backend} ${entry.kind} ---`);
  if (entry.metadata) {
    const { numInputs, numOutputs, ...rest } = entry.metadata;
    const parts = [`in=${numInputs}`, `out=${numOutputs}`];
    for (const [k, v] of Object.entries(rest)) {
      if (
        typeof v === "string" ||
        typeof v === "number" ||
        typeof v === "boolean"
      )
        parts.push(k + "=" + String(v));
    }
    console.log(parts.join("  "));
  }
  if (entry.code) {
    console.log(entry.code);
  }
}

console.log("\nResult:", await result.jsAsync());
