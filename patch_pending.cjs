const fs = require("fs");
let code = fs.readFileSync("src/backend/webgpu.ts", "utf8");

const classCode = `
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
           // Async compilation
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

    this.backend._executeCommandTapeDeferred(this.tape, this.inputSlots, this.outputSlots, this.outputTableIdxs);
  }
}
`;

code = code.replace(
  "export class WebGPUBackend implements Backend {",
  classCode + "\nexport class WebGPUBackend implements Backend {",
);
fs.writeFileSync("src/backend/webgpu.ts", code);
