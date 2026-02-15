import type { Device, numpy as np } from "@jax-js-nonconsuming/jax";
import type { Plugin } from "@rollup/browser";

import { arrayToDataUrl } from "./displayImage";

export type ConsoleLine = {
  level: "log" | "info" | "warn" | "error" | "image";
  data: string[];
  time: number;
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
    const startedRunAt = performance.now();
    try {
      const result = await _runProgram(source, device, this);
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
): Promise<RunResult> {
  const [jax, optax, loaders] = await Promise.all([
    import("@jax-js-nonconsuming/jax"),
    import("@jax-js-nonconsuming/optax"),
    import("@jax-js-nonconsuming/loaders"),
  ]);
  const ts = await import("typescript");
  const { rollup } = await import("@rollup/browser");

  const mockConsole = runner.mockConsole;

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
        return ts.transpileModule(code, {
          compilerOptions: {
            module: ts.ModuleKind.ESNext,
            target: ts.ScriptTarget.ES2022,
          },
        }).outputText;
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
        "@jax-js-nonconsuming/jax",
        "@jax-js-nonconsuming/optax",
        "@jax-js-nonconsuming/loaders",
      ],
    });

    // We use the "system" format because it allows you to use async/await.
    // https://rollupjs.org/repl/
    const { output } = await bundle.generate({
      file: "bundle.js",
      format: "system",
    });

    const header = `
      const console = _BUILTINS.console;
      const displayImage = _BUILTINS.displayImage;

      const System = { register(externals, f) {
        const { execute, setters } = f();
        for (let i = 0; i < externals.length; i++) {
          setters[i](_MODULES[externals[i]]);
        }
        this.f = execute;
      } };`;
    const trailer = `;await (async () => System.f())()`;
    const bundledCode = header + output[0].code + trailer;

    // AsyncFunction constructor, analogous to Function.
    const AsyncFunction: typeof Function = async function () {}
      .constructor as any;

    const startTime = performance.now();
    await new AsyncFunction("_MODULES", "_BUILTINS", bundledCode)(
      // _MODULES
      {
        "@jax-js-nonconsuming/jax": jax,
        "@jax-js-nonconsuming/optax": optax,
        "@jax-js-nonconsuming/loaders": loaders,
      },
      // _BUILTINS
      {
        console: mockConsole,
        displayImage: displayImage,
      },
    );
    return {
      success: true,
      duration: performance.now() - startTime,
    };
  } catch (e: any) {
    mockConsole.error(e);
    return { success: false, duration: 0 };
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
