const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

// I need to add IPendingExecute to imports from array.ts
code = code.replace(/import \{ SyncReader \} from "\.\.\/backend\.js";/, 'import { SyncReader } from "../backend.js";\nimport type { IPendingExecute } from "../frontend/array.js";');

const classDef = `class PendingCommandTape implements IPendingExecute {
  prepared: any = null;
  submitted = false;
  #promise: Promise<void> | null = null;
  #rc = 1;

  constructor(
    readonly backend: WebGPUBackend,
    readonly tape: WebGPUCommandTape,
    readonly buffers: GPUBuffer[],
    readonly dynBindSizes: number[],
    readonly dynOffsets: number[]
  ) {}

  updateRc(delta: number) {
    if (this.#rc <= 0) throw new Error("internal: PendingCommandTape used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      // It was canceled. The buffers created by executeCommandTape were already placed into slots.
      // Wait, executeCommandTape generated Slots which belong to Arrays, they will be freed by JS eventually natively. Nothing to do.
    }
  }

  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      // Map over all dispatches and ensure pipelines are compiled asynchronously
      const promises: Promise<void>[] = [];
      for (const op of this.tape) {
        if (op.type === "dispatch" || op.type === "scatter_add" || op.type === "reverse" || op.type === "dus") {
           // We need to resolve the pipelines inside them.
           // scatter_add, reverse, dus pipelines are already synchronous via generic caches?
           // Actually, only "dispatch" operations use customized shaders in the JIT step!
        }
        if (op.type === "dispatch") {
          promises.push(this.backend.pipelines.prepare(op.dispatch.shader).then(p => {
             // attach it temporarily or we resolve it here?
             // Since TapeDispatch doesn't have pipeline anymore, we can just look it up via prepareSync later during submit
          }));
        } else if (op.type === "scan") {
             // Can we do anything?
        }
      }
      await Promise.all(promises);
      this.prepared = true;
    })();
    await this.#promise;
  }

  prepareSync() {
    if (this.prepared) return;
    for (const op of this.tape) {
      if (op.type === "dispatch") {
        this.backend.pipelines.prepareSync(op.dispatch.shader);
      }
    }
    this.prepared = true;
  }

  submit() {
    if (this.submitted) return;
    if (this.#rc <= 0) throw new Error("internal: PendingCommandTape used rc<=0");
    this.prepareSync(); // Ensure done
    this.submitted = true;
    
    // We execute the actual WebGPU commands here!
    this.backend.submitCommandTape(this.tape, this.buffers, this.dynBindSizes, this.dynOffsets);
  }
}
`;

code = code.replace(
  '  executeCommandTape(',
  `${classDef}\n  executeCommandTape(`
);

// We need to replace the body of executeCommandTape to just construct buffers, but DEFER the command generation!
// Let's refactor executeCommandTape directly.

fs.writeFileSync('src/backend/webgpu.ts', code);
