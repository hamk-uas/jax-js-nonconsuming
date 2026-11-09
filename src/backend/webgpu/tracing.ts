import { Kernel } from "../../alu";
import { Routine } from "../../routine";
import {
  emitTrace,
  isTracing,
  type TraceInfo,
  traceSourceInfo,
} from "../../tracing";

const MAX_TIMESTAMP_QUERIES = 4096;

interface TracingEntry extends TraceInfo {
  beginIndex: number;
  endIndex: number;
}

interface TracingBatch {
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  dst: GPUBuffer;
  nextIndex: number;
  entries: TracingEntry[];
}

const encoderBatches = new WeakMap<GPUCommandEncoder, TracingBatch[]>();

function createTracingBatch(device: GPUDevice): TracingBatch {
  return {
    querySet: device.createQuerySet({
      type: "timestamp",
      count: MAX_TIMESTAMP_QUERIES,
    }),
    resolve: device.createBuffer({
      size: MAX_TIMESTAMP_QUERIES * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    dst: device.createBuffer({
      size: MAX_TIMESTAMP_QUERIES * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
    nextIndex: 0,
    entries: [],
  };
}

export interface TracingSlot {
  batch: TracingBatch;
  beginIndex: number;
  endIndex: number;
}

function acquireTracingSlot(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
): TracingSlot | undefined {
  if (!device.features.has("timestamp-query")) return undefined;

  let batches = encoderBatches.get(encoder);
  if (!batches) encoderBatches.set(encoder, (batches = []));
  let batch = batches.at(-1);
  if (!batch || batch.nextIndex >= MAX_TIMESTAMP_QUERIES) {
    batch = createTracingBatch(device);
    batches.push(batch);
  }

  const beginIndex = batch.nextIndex;
  const endIndex = beginIndex + 1;
  batch.nextIndex += 2;
  return { batch, beginIndex, endIndex };
}

/**
 * If tracing is active, acquire a slot for timestamp queries.
 *
 * Returns undefined if tracing is not active or the device doesn't support
 * timestamp queries.
 */
export function maybeAcquireTracingSlot(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
): TracingSlot | undefined {
  if (!isTracing()) return undefined;
  return acquireTracingSlot(device, encoder);
}

/** Resolve timestamps in the encoder that wrote them, then submit and read back. */
export function submitWithTracing(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
): void {
  const batches = encoderBatches.get(encoder);
  if (!batches) {
    device.queue.submit([encoder.finish()]);
    return;
  }
  let submitted = false;
  try {
    // Intel/Dawn produced partially converted timestamps with a separate
    // resolve submission. Keeping writes and resolves together also prevents
    // stopTrace() or query rollover from resolving unsubmitted writes.
    for (const batch of batches) {
      if (batch.entries.length === 0) continue;
      encoder.resolveQuerySet(
        batch.querySet,
        0,
        batch.nextIndex,
        batch.resolve,
        0,
      );
      encoder.copyBufferToBuffer(
        batch.resolve,
        0,
        batch.dst,
        0,
        batch.nextIndex * 8,
      );
    }
    device.queue.submit([encoder.finish()]);
    submitted = true;
  } finally {
    if (!submitted) discardTracingEncoder(encoder);
  }
  encoderBatches.delete(encoder);
  for (const batch of batches) {
    if (batch.entries.length > 0) readTracingBatch(batch);
    else destroyTracingBatch(batch);
  }
}

/** Release trace resources for an encoder abandoned on an encoding error. */
export function discardTracingEncoder(encoder: GPUCommandEncoder): void {
  const batches = encoderBatches.get(encoder);
  if (!batches) return;
  encoderBatches.delete(encoder);
  for (const batch of batches) destroyTracingBatch(batch);
}

/** Only the first and last passes of a dispatch write its timestamps. */
export function tracingTimestampWrites(
  slot: TracingSlot | undefined,
  passIndex: number,
  numPasses: number,
): GPUComputePassTimestampWrites | undefined {
  if (!slot || (passIndex !== 0 && passIndex !== numPasses - 1))
    return undefined;
  return {
    querySet: slot.batch.querySet,
    beginningOfPassWriteIndex: passIndex === 0 ? slot.beginIndex : undefined,
    endOfPassWriteIndex:
      passIndex === numPasses - 1 ? slot.endIndex : undefined,
  };
}

/**
 * Record a tracing entry for a pipeline dispatch.
 */
export function recordTrace(
  slot: TracingSlot,
  source: Kernel | Routine,
  numPasses: number,
  wgslSource: string,
): void {
  const info = traceSourceInfo(source);
  info.properties.push(["passes", `${numPasses}`]);
  info.properties.push(["source", wgslSource]);
  slot.batch.entries.push({
    ...info,
    beginIndex: slot.beginIndex,
    endIndex: slot.endIndex,
  });
}

function readTracingBatch(batch: TracingBatch): void {
  const { entries } = batch;
  batch.dst.mapAsync(GPUMapMode.READ).then(
    () => {
      try {
        const times = new BigInt64Array(batch.dst.getMappedRange());

        // Anchor this batch's last GPU timestamp to the CPU clock at readback;
        // browsers add jitter to timestamp queries, so they are only usable
        // relative to each other.
        const anchorGpuNs = times[entries[entries.length - 1].endIndex];
        const anchorCpuMs = performance.now();

        for (const entry of entries) {
          const startMs =
            anchorCpuMs + Number(times[entry.beginIndex] - anchorGpuNs) / 1e6;
          const endMs =
            anchorCpuMs + Number(times[entry.endIndex] - anchorGpuNs) / 1e6;
          emitTrace("webgpu", entry, startMs, endMs);
        }
      } finally {
        batch.dst.unmap();
        destroyTracingBatch(batch);
      }
    },
    (error: unknown) => {
      destroyTracingBatch(batch);
      console.error("Failed to read WebGPU trace timestamps:", error);
    },
  );
}

function destroyTracingBatch(batch: TracingBatch): void {
  batch.querySet.destroy();
  batch.resolve.destroy();
  batch.dst.destroy();
}
