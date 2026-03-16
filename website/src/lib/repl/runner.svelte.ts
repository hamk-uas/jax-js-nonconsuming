import type {
  CodeCaptureEntry,
  Device,
  numpy as np,
} from "@hamk-uas/jax-js-nonconsuming";
import type { Plugin } from "@rollup/browser";

import { arrayToDataUrl } from "./displayImage";
import {
  type LeakMarker,
  type SourceMapLike,
  parseLeakMarkers,
  remapLeakDetails,
  remapReplLocationText,
} from "./sourcemap";

export type { LeakMarker };

export type ConsoleLine = {
  level: "log" | "info" | "warn" | "error" | "image" | "code";
  data: string[];
  time: number;
  codeEntry?: CodeCaptureEntry;
};

// Intercepted methods similar to console.log().
const consoleMethods = [
  "clear",
  "error",
  "info",
  "log",
  "time",
  "timeEnd",
  "timeLog",
  "trace",
  "warn",
] as const;

export class ReplRunner {
  running = $state(false);
  finished = $state(false);
  consoleLines: ConsoleLine[] = $state([]);
  runDurationMs = $state<number | null>(null);
  detailedLeakDiagnostics = $state(true);
  captureCode = $state(false);
  leakMarkers: LeakMarker[] = $state([]);
  consoleTimers = new Map<string, number>();
  mockConsole: Console;

  constructor() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const runner = this;

    this.mockConsole = new Proxy(console, {
      get(target, prop, receiver) {
        if (consoleMethods.some((m) => m === prop)) {
          return (...args: any[]) => {
            runner.#handleMockConsole(prop as any, ...args);
            Reflect.get(target, prop, receiver)(...args);
          };
        }
        return Reflect.get(target, prop, receiver);
      },
    });
  }

  async runProgram(source: string, device: Device) {
    if (this.running) return;
    this.running = true;
    this.finished = false;
    this.runDurationMs = null;
    this.leakMarkers = [];
    const startedRunAt = performance.now();
    try {
      const result = await _runProgram(
        source,
        device,
        this,
        this.detailedLeakDiagnostics,
      );
      const endedRunAt = performance.now();
      if (endedRunAt - startedRunAt < 100) {
        // Take at least 100ms, otherwise it's unclear it actually ran.
        await new Promise((resolve) =>
          setTimeout(resolve, 100 - (endedRunAt - startedRunAt)),
        );
      }
      if (result.success) {
        this.runDurationMs = result.duration;
      }
    } finally {
      this.running = false;
      this.finished = true;
    }
  }

  #handleMockConsole(method: (typeof consoleMethods)[number], ...args: any[]) {
    if (
      method === "log" ||
      method === "info" ||
      method === "warn" ||
      method === "error"
    ) {
      this.consoleLines.push({
        level: method,
        data: args.map((x) =>
          typeof x === "string"
            ? x
            : x instanceof Error
              ? x.toString()
              : formatObject(x),
        ),
        time: Date.now(),
      });
    } else if (method === "clear") {
      this.consoleLines = [];
    } else if (method === "trace") {
      this.consoleLines.push({
        level: "error",
        data: ["Received stack trace, see console for details."],
        time: Date.now(),
      });
    } else if (method === "time") {
      this.consoleTimers.set(args[0], performance.now());
    } else if (method === "timeLog") {
      const start = this.consoleTimers.get(args[0]);
      if (start !== undefined) {
        const elapsed = performance.now() - start;
        this.consoleLines.push({
          level: "log",
          data: [`${args[0]}: ${elapsed.toFixed(1)}ms`],
          time: Date.now(),
        });
      }
    } else if (method === "timeEnd") {
      const start = this.consoleTimers.get(args[0]);
      if (start !== undefined) {
        const elapsed = performance.now() - start;
        this.consoleLines.push({
          level: "log",
          data: [`${args[0]}: ${elapsed.toFixed(1)}ms - timer ended`],
          time: Date.now(),
        });
        this.consoleTimers.delete(args[0]);
      }
    }
  }
}

interface RunResult {
  success: boolean;
  duration: number;
}

async function _runProgram(
  source: string,
  device: Device,
  runner: ReplRunner,
  detailedLeakDiagnostics: boolean,
): Promise<RunResult> {
  const [jax, optax, loaders] = await Promise.all([
    import("@hamk-uas/jax-js-nonconsuming"),
    import("@hamk-uas/jax-js-nonconsuming-optax"),
    import("@hamk-uas/jax-js-nonconsuming-loaders"),
  ]);
  const ts = await import("typescript");
  const { rollup } = await import("@rollup/browser");

  const mockConsole = runner.mockConsole;
  const getSlotCount = (): number | null => {
    try {
      const backend = (jax as any).getBackend?.();
      if (backend && typeof backend.slotCount === "function") {
        return backend.slotCount();
      }
    } catch {
      // Ignore leak diagnostics errors in REPL.
    }
    return null;
  };
  const slotsBefore = getSlotCount();
  let detailedLeakTrackingStarted = false;
  let detailedLeakSummary: string | null = null;
  let detailedLeakCount = 0;
  let generatedSourceMap: SourceMapLike | null = null;

  // Builtins for the REPL environment.
  const np = jax.numpy;
  const displayImage = async (ar: np.Array) => {
    const dataUrl = await arrayToDataUrl(ar);
    runner.consoleLines.push({
      level: "image",
      data: [dataUrl],
      time: Date.now(),
    });
  };

  mockConsole.clear();

  const devices = await jax.init();
  if (devices.includes(device)) {
    jax.defaultDevice(device);
  } else {
    mockConsole.warn(`${device} not supported, using wasm`);
    jax.defaultDevice("wasm");
  }

  // Create a simple virtual module plugin to resolve our in-memory modules.
  const virtualPlugin: Plugin = {
    name: "virtual",
    resolveId(id) {
      // We treat 'index.ts' as the user code entry point.
      if (id === "index.ts") {
        return id;
      } else {
        throw new Error("Module not found: " + id);
      }
    },
    load(id) {
      if (id === "index.ts") {
        return source;
      } else {
        return null;
      }
    },
  };

  const typescriptPlugin: Plugin = {
    name: "typescript",
    transform(code, id) {
      if (id.endsWith(".ts")) {
        const result = ts.transpileModule(code, {
          compilerOptions: {
            module: ts.ModuleKind.ESNext,
            // ESNext emits `using` declarations natively (just strips types).
            // ES2022 downleveled `using` into 80+ lines of try/catch/finally
            // helper code whose source map had gaps, breaking leak diagnostic
            // line remapping. All browsers with WebGPU (REPL prerequisite)
            // support `using` natively (Chrome 134+, Firefox 134+, Safari 18.2+).
            target: ts.ScriptTarget.ESNext,
            sourceMap: true,
          },
          fileName: id,
        });
        // Strip the //# sourceMappingURL from outputText so it doesn't
        // leak into the Rollup output and confuse V8 into trying to load
        // a non-existent index.js.map file.
        const strippedCode = result.outputText.replace(
          /\n?\/\/# sourceMappingURL=[^\n]*/,
          "",
        );
        return { code: strippedCode, map: result.sourceMapText };
      }
      return null;
    },
  };

  try {
    // Use @rollup/browser to bundle the code.
    const bundle = await rollup({
      input: "index.ts",
      plugins: [typescriptPlugin, virtualPlugin],
      external: [
        "@hamk-uas/jax-js-nonconsuming",
        "@hamk-uas/jax-js-nonconsuming-optax",
        "@hamk-uas/jax-js-nonconsuming-loaders",
      ],
    });

    // We use the "system" format because it allows you to use async/await.
    // Source map is "hidden" — we collect it but position it ourselves.
    // https://rollupjs.org/repl/
    const { output } = await bundle.generate({
      file: "bundle.js",
      format: "system",
      sourcemap: "hidden",
    });

    // Single-line header so the source map is only off by 1 line.
    const header =
      "const console=_BUILTINS.console,displayImage=_BUILTINS.displayImage," +
      "System={register(e,f){const{execute:x,setters:s}=f();" +
      "for(let i=0;i<e.length;i++)s[i](_MODULES[e[i]]);this.f=x}};";
    const trailer = ";await(async()=>System.f())()";

    // Prepare the source map for manual remapping of leak diagnostics.
    // We do NOT embed the source map in the bundled code via
    // //# sourceMappingURL because Chrome 132+ applies source maps to
    // Error.stack automatically. If we embedded it, checkLeaks positions
    // would be double-mapped (V8 maps once, our remapLeakDetails maps
    // again), producing wrong line numbers. Instead we keep the map
    // only for our own manual remapping in remapLeakDetails/parseLeakMarkers.
    const map = output[0].map;
    generatedSourceMap = map as SourceMapLike;
    if (map && map.mappings) {
      // Prepend ";;;" to shift all mappings down by 3 generated lines:
      //   +2 lines for the AsyncFunction wrapper that V8 adds
      //      (line 1: `async function anonymous(_MODULES,_BUILTINS`, line 2: `) {`)
      //   +1 line for the single-line header we prepend to bundledCode
      // V8 reports Error.stack positions relative to the full AsyncFunction
      // source (including the wrapper lines), so the source map must account
      // for all 3 extra lines before the Rollup output begins.
      map.mappings = ";;;" + map.mappings;
    }

    const bundledCode =
      header +
      "\n" +
      output[0].code +
      "\n" +
      trailer +
      "\n//# sourceURL=index.ts";

    // AsyncFunction constructor, analogous to Function.
    const AsyncFunction: typeof Function = async function () {}
      .constructor as any;

    const startTime = performance.now();
    if (detailedLeakDiagnostics && jax.checkLeaks?.start) {
      try {
        jax.checkLeaks.start();
        detailedLeakTrackingStarted = true;
      } catch {
        // Fall back to lightweight slot diagnostics.
      }
    }

    if (runner.captureCode) {
      jax.setCodeCapture((entry) => {
        runner.consoleLines.push({
          level: "code",
          data: [],
          time: Date.now(),
          codeEntry: entry,
        });
      });
    }

    await new AsyncFunction("_MODULES", "_BUILTINS", bundledCode)(
      // _MODULES
      {
        "@hamk-uas/jax-js-nonconsuming": jax,
        "@hamk-uas/jax-js-nonconsuming-optax": optax,
        "@hamk-uas/jax-js-nonconsuming-loaders": loaders,
      },
      // _BUILTINS
      {
        console: mockConsole,
        displayImage: displayImage,
      },
    );

    if (detailedLeakTrackingStarted && jax.checkLeaks?.stop) {
      try {
        const report = jax.checkLeaks.stop();
        const remappedDetails = remapLeakDetails(
          report.details,
          generatedSourceMap,
        );
        detailedLeakSummary = remapReplLocationText(
          report.summary,
          generatedSourceMap,
        );
        detailedLeakCount = report.leaked;
        runner.leakMarkers = parseLeakMarkers(remappedDetails);
      } catch {
        // Ignore checkLeaks reporting failures in REPL.
      }
    }

    // Use checkLeaks report as the authoritative leak count when available —
    // its baseline is captured at start() (after init), so it's reliable
    // across repeated runs. The slot-count diff (slotsBefore vs slotsAfter)
    // is unreliable because slotsBefore is captured before init() and
    // _disposeAllJitCaches() in stop() can free slots from prior runs.
    if (detailedLeakCount > 0 && detailedLeakSummary) {
      mockConsole.warn(
        `REPL note: ${detailedLeakCount} array slot(s) still live after this run. Detailed leak diagnostics:`,
      );
      mockConsole.warn(detailedLeakSummary);
    } else if (!detailedLeakTrackingStarted) {
      const slotsAfter = getSlotCount();
      if (
        slotsBefore !== null &&
        slotsAfter !== null &&
        slotsAfter > slotsBefore
      ) {
        const leaked = slotsAfter - slotsBefore;
        mockConsole.warn(
          `REPL note: ${leaked} array slot(s) still live after this run. Use using declarations or .dispose() for arrays you create. Enable Detailed leak diagnostics and run again to see leak origins.`,
        );
      }
    }
    return {
      success: true,
      duration: performance.now() - startTime,
    };
  } catch (e: any) {
    if (detailedLeakTrackingStarted && jax.checkLeaks?.stop) {
      try {
        const report = jax.checkLeaks.stop();
        const remappedDetails = remapLeakDetails(
          report.details,
          generatedSourceMap,
        );
        detailedLeakSummary = remapReplLocationText(
          report.summary,
          generatedSourceMap,
        );
        detailedLeakCount = report.leaked;
        runner.leakMarkers = parseLeakMarkers(remappedDetails);
      } catch {
        // Ignore checkLeaks reporting failures in REPL.
      }
    }

    if (detailedLeakCount > 0 && detailedLeakSummary) {
      mockConsole.warn(
        `REPL note: ${detailedLeakCount} array slot(s) still live after this run. Detailed leak diagnostics:`,
      );
      mockConsole.warn(detailedLeakSummary);
    } else if (!detailedLeakTrackingStarted) {
      const slotsAfter = getSlotCount();
      if (
        slotsBefore !== null &&
        slotsAfter !== null &&
        slotsAfter > slotsBefore
      ) {
        const leaked = slotsAfter - slotsBefore;
        mockConsole.warn(
          `REPL note: ${leaked} array slot(s) still live after this run. Use using declarations or .dispose() for arrays you create. Enable Detailed leak diagnostics and run again to see leak origins.`,
        );
      }
    }
    mockConsole.error(e);
    return { success: false, duration: 0 };
  } finally {
    if (runner.captureCode) jax.setCodeCapture(null);
  }
}

function formatObject(obj: any): string {
  const buffer: string[] = [];
  obj = _convertJaxArrays(obj);
  _formatObject(obj, "", buffer);
  return buffer.join("");
}

function _convertJaxArrays(obj: any): any {
  if (typeof obj !== "object" || obj === null) {
    return obj;
  } else if (typeof obj["js"] === "function") {
    return obj.js();
  } else if (Array.isArray(obj)) {
    return obj.map((x) => _convertJaxArrays(x));
  } else {
    const newObj: any = {};
    for (const [k, v] of Object.entries(obj)) {
      newObj[k] = _convertJaxArrays(v);
    }
    return newObj;
  }
}

/**
 * Format an object with indentation, in JSON style, and keeping lists / objects
 * inline if they don't exceed 120 characters in width.
 */
function _formatObject(obj: any, indent: string, buffer: string[]) {
  if (typeof obj !== "object" || obj === null) {
    buffer.push(_stringifyOneLine(obj));
    return;
  }

  const strRep = _stringifyOneLine(obj);
  if (strRep.length <= 120) {
    buffer.push(strRep);
  } else {
    if (Array.isArray(obj)) {
      buffer.push("[\n");
      const newIndent = indent + "  ";
      for (let i = 0; i < obj.length; i++) {
        buffer.push(newIndent);
        _formatObject(obj[i], newIndent, buffer);
        buffer.push(",\n");
      }
      buffer.push(indent + "]");
    } else {
      buffer.push("{\n");
      const newIndent = indent + "  ";
      const keys = Object.keys(obj);
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        buffer.push(newIndent + _stringifyKey(key) + ": ");
        _formatObject(obj[key], newIndent, buffer);
        buffer.push(",\n");
      }
      buffer.push(indent + "}");
    }
  }
}

function _stringifyOneLine(obj: any): string {
  if (typeof obj === "number") {
    // Format numbers with up to 7 significant digits.
    return obj.toPrecision(7).replace(/\.?0+$/, "");
  } else if (typeof obj !== "object" || obj === null) {
    return JSON.stringify(obj);
  } else if (Array.isArray(obj)) {
    return "[" + obj.map(_stringifyOneLine).join(", ") + "]";
  } else {
    return (
      "{ " +
      Object.entries(obj)
        .map(([k, v]) => `${_stringifyKey(k)}: ${_stringifyOneLine(v)}`)
        .join(", ") +
      " }"
    );
  }
}

function _stringifyKey(key: string): string {
  // If key is a valid identifier, return as is.
  if (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(key)) {
    return key;
  }
  return JSON.stringify(key);
}
