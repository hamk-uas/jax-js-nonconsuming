import type { JitReport } from "@hamk-uas/jax-js-nonconsuming";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

declare global {
  function displayImage(param: np.Array): Promise<void>;
  function captureJitReport(
    fn: (...args: any[]) => any,
    ...args: any[]
  ): Promise<JitReport>;
  function formatJitReport(report: JitReport): string;
}

export {};
