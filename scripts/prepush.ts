import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

import { detectGpus, type GpuProfile } from "../test/gpu-config";

export function runPrepush(gpus: GpuProfile[] = detectGpus()): number {
  if (gpus.length === 0) {
    console.warn(
      "[pre-push] No supported GPU detected. Running CPU/WASM only; " +
        "GPU coverage is NOT validated.",
    );
  }

  const commands = [
    {
      command: "pnpm",
      args: gpus.length > 0 ? ["vitest", "run"] : ["run", "test:cpu"],
      gpu: undefined,
    },
    ...gpus.map((gpu) => ({
      command: "bash",
      args: ["scripts/gpu-test.sh", "run"],
      gpu,
    })),
  ];
  let exitCode = 0;
  for (const { command, args, gpu } of commands) {
    console.log(`[pre-push] ${gpu ?? "default"}: ${command} ${args.join(" ")}`);
    const result = spawnSync(command, args, {
      stdio: "inherit",
      env: {
        ...process.env,
        JAX_TEST_CPU_ONLY: "0",
        JAX_REQUIRE_DEVICES: gpus.length > 0 ? "webgpu,webgl" : "cpu,wasm",
        ...(gpu ? { GPU: gpu } : {}),
      },
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
      console.error(
        `[pre-push] ${gpu ?? "default"} failed: ` +
          (result.signal ? `signal ${result.signal}` : `exit ${result.status}`),
      );
      exitCode = result.status ?? 1;
    }
  }
  return exitCode;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  process.exitCode = runPrepush();
}
