import type { BackendCapabilities } from "../src/backend";
import { CostFeatures, evaluateTotalCost } from "../src/tuner";

const capsIntelOld: Partial<BackendCapabilities> = {
  dispatchOverheadUs: 25,
  bandwidthGBs: 30,
  tflops: 0.4,
  adapterVendor: "intel",
  adapterArchitecture: "gen_12lp",
  inferredVendorClass: "igp",
  maxComputeWorkgroupSizeX: 256,
};

const baseCandidates = [
  { Br: 64, Bc: 64, Bk: 16, threadTile: [8, 8] },
  { Br: 64, Bc: 64, Bk: 16, threadTile: [4, 4] },
  { Br: 32, Bc: 32, Bk: 16, threadTile: [4, 4] },
  { Br: 32, Bc: 32, Bk: 16, threadTile: [2, 2] },
  { Br: 16, Bc: 16, Bk: 16, threadTile: [1, 1] },
];

function testFixedEvaluate(
  features: CostFeatures,
  caps: Partial<BackendCapabilities>,
) {
  return evaluateTotalCost(features, caps);
}

function evalCaps(caps: Partial<BackendCapabilities>) {
  console.log(
    `\nEvaluating for ${caps.adapterVendor} (${caps.tflops} TFLOPS) [arch: ${caps.adapterArchitecture}]`,
  );
  for (const c of baseCandidates) {
    const bytesPerElem = 4;
    const tt = c.threadTile;
    const numThreads = (c.Br / tt[0]) * (c.Bc / tt[1]);
    const tileA = c.Br * c.Bk * bytesPerElem;
    const tileB = c.Bk * c.Bc * bytesPerElem;
    const padOverhead = Math.ceil((tileA + tileB) * 0.07);
    const shmemBytes = tileA + tileB + padOverhead;
    const wgslSize = 8192 + tt[0] * tt[1] * 100;

    // Explicit depth Priv matched to previous legacy:
    // For 4x4 tile where Bk=16, previous comment: 64x64 tt=4,4 takes 16 registers mapped manually
    // Wait, the equation in lax.ts: depthPriv: (tt[0] * tt[1]) + c.Bk * 2 + 10
    // for tt=4,4 it is 16 + 32 + 10 = 58 registers. 256 threads * 58 registers = 14,848 registers!
    const features: CostFeatures = {
      nDispatch: 1,
      nBuffers: 3,
      countAlu: c.Br * c.Bc * (2 * c.Bk),
      countMem: (c.Br * c.Bk + c.Bk * c.Bc + c.Br * c.Bc) * bytesPerElem,
      depthPriv: tt[0] * tt[1] + c.Bk * 2 + 10,
      sizeShmem: shmemBytes,
      sizeWgsl: wgslSize,
      parallelism: numThreads,
      produceCount: c.Br * c.Bc,
    };

    const cost = testFixedEvaluate(features, caps);
    console.log(
      `Br=${c.Br} Bc=${c.Bc} tt=${tt} -> cost=${cost.toExponential(4)}, threads=${numThreads}, dp=${features.depthPriv}, totR=${features.depthPriv * numThreads}`,
    );
  }
}

evalCaps(capsIntelOld);
