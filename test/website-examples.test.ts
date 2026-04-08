import {
  clearCaches,
  defaultDevice,
  init,
} from "@hamk-uas/jax-js-nonconsuming";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

const websiteExamples = [
  {
    id: "homepage",
    device: "wasm" as const,
    load: () => import("../website/src/lib/repl/examples/homepage"),
  },
  {
    id: "01-arrays",
    device: "wasm" as const,
    load: () => import("../website/src/routes/repl/01-arrays"),
  },
  {
    id: "02-tracing",
    device: "wasm" as const,
    load: () => import("../website/src/routes/repl/02-tracing"),
  },
  {
    id: "03-logistic-regression",
    device: "wasm" as const,
    load: () => import("../website/src/routes/repl/03-logistic-regression"),
  },
  {
    id: "04-mandelbrot",
    device: "webgpu" as const,
    load: () => import("../website/src/routes/repl/04-mandelbrot"),
  },
];

beforeAll(async () => {
  (globalThis as any).displayImage = async () => {};
});

beforeEach(() => {
  clearCaches();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("website examples", () => {
  for (const example of websiteExamples) {
    it(`${example.id} imports and runs`, { timeout: 30_000 }, async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      const devices = await init(example.device);
      if (!devices.includes(example.device)) {
        return;
      }
      defaultDevice(example.device);
      await expect(example.load()).resolves.toBeDefined();

      const rescueWarnings = warnSpy.mock.calls
        .map((args) => String(args[0] ?? ""))
        .filter((msg) => msg.includes("would leak in eager mode"));
      const gradRescues = rescueWarnings.filter((msg) =>
        msg.includes("grad/vjp rescued"),
      );

      expect(gradRescues).toEqual([]);
    });
  }
});
