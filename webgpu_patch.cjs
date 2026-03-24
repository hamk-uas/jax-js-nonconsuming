const fs = require("fs");

let webgpu = fs.readFileSync("src/backend/webgpu.ts", "utf8");

// We will inject PendingCommandTape right before WebGPUBackend!

const pendingCode = `
class PendingCommandTape implements IPendingExecute {
  prepared: any = null;
  submitted = false;
  #promise: Promise<void> | null = null;
  #rc = 1;

  constructor(
    readonly backend: WebGPUBackend,
    readonly tape: WebGPUCommandTape,
    readonly inputSlots: Slot[],
    readonly outputSlots: Slot[],
    readonly outputTableIdxs: number[]
  ) {}

  updateRc(delta: number) {
    if (this.#rc <= 0) throw new Error("internal: PendingCommandTape used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      // It was canceled. The Slots generated synchronously will be freed by Array GC.
    }
  }

  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      const promises: Promise<any>[] = [];
      for (const op of this.tape.ops) {
        if (op.type === "dispatch") {
           // We must prepare the shader asynchronously
           promises.push(this.backend.pipelines.prepare(op.dispatch.shader));
        } else if (op.type === "scan") {
            const scan = op.scan;
            if (scan.mode === "compiled-loop" && scan.nativeShader) promises.push(this.backend.pipelines.prepare(scan.nativeShader));
            if (scan.mode === "preencoded-multi-step") {
                if (scan.passShaders) scan.passShaders.forEach(s => promises.push(this.backend.pipelines.prepare(s)));
                if (scan.stepPipelines) scan.stepPipelines.forEach(s => promises.push(this.backend.pipelines.prepare(s.shader)));
                if (scan.yStackPipeline) promises.push(this.backend.pipelines.prepare(scan.yStackPipeline.shader));
                if (scan.reductionPipeline) promises.push(this.backend.pipelines.prepare(scan.reductionPipeline.shader));
            }
            if (scan.mode === "preencoded-routine") {
                if (scan.passShaders) scan.passShaders.forEach(s => promises.push(this.backend.pipelines.prepare(s)));
            }
        } else if (op.type === "scatter_add") {
            // scatter pipeline should be prepared if possible, but actually we use dynamic caching there normally.
            // Oh wait, scatter uses getScatterPipeline which is synchronous caching but might compile.
            // It's ok if scatter blocks briefly or if we don't await it here.
        }
      }
      await Promise.all(promises);
      this.prepared = true;
    })();
    return this.#promise;
  }

  prepareSync() {}

  submit() {
    if (this.submitted) return;
    this.submitted = true;
    if (this.#rc <= 0) return;

    // Full execution of the tape natively!
    this.backend._executeCommandTapeDeferred(this.tape, this.inputSlots, this.outputSlots, this.outputTableIdxs);
  }
}
`;

webgpu = webgpu.replace(
  "export class WebGPUBackend implements Backend {",
  pendingCode + "\nexport class WebGPUBackend implements Backend {",
);
fs.writeFileSync("src/backend/webgpu.ts", webgpu);
