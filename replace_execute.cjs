const fs = require('fs');

let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

// I will find executeCommandTape completely and rewrite it:
const searchStart = "  executeCommandTape(tape: WebGPUCommandTape, inputSlots: Slot[]): Slot[] {";

const replacementCode = `  executeCommandTape(tape: WebGPUCommandTape, inputSlots: Slot[]): { outputs: Slot[]; pending: IPendingExecute[] } {
    // We execute synchronously only what is strictly necessary to return valid Slots!
    // We need to calculate sizes accurately without running the allocations.
    
    // Fortunately, compileCommandTape recorded outputSizes on WebGPUCommandTape (if we patched it).
    // Let's rely on that!
    if (!tape.outputSizes) {
      tape.outputSizes = new globalThis.Array(tape.outputTableIdxs.length);
      // fallback calculation if not patched
      const knownSizes = new Map<number, number>();
      for (let i = 0; i < inputSlots.length; i++) {
        const { size } = this.buffers.get(inputSlots[i])!;
        knownSizes.set(tape.inputTableIdxs[i], size);
      }
      if (tape.constSlab) {
        for (const e of tape.constSlab.entries) knownSizes.set(e.tableIdx, e.originalSize);
      }
      if (tape.arenaSlabs) {
        for (const slab of tape.arenaSlabs) {
          for (const e of slab.entries) knownSizes.set(e.tableIdx, e.originalSize);
        }
      }
      for (const op of tape.ops) {
        if (op.type === "malloc") knownSizes.set(op.malloc.tableIdx, op.malloc.originalSize);
        if (op.type === "recycle") knownSizes.set(op.toIdx, knownSizes.get(op.fromIdx)!);
      }
      for (let i = 0; i < tape.outputTableIdxs.length; i++) {
        tape.outputSizes[i] = knownSizes.get(tape.outputTableIdxs[i])!;
      }
    }

    const outputSlots: Slot[] = new globalThis.Array(tape.outputTableIdxs.length);
    for (let i = 0; i < tape.outputTableIdxs.length; i++) {
      const size = tape.outputSizes[i];
      const slot = this.nextSlot++;
      // We push a Slot with undefined buffer — it will be populated by PendingCommandTape.submit()
      this.buffers.set(slot, { buffer: undefined as any as GPUBuffer, size, ref: 1 });
      outputSlots[i] = slot;
    }

    const pending = new PendingCommandTape(this, tape, inputSlots, outputSlots, tape.outputTableIdxs);
    return { outputs: outputSlots, pending: [pending] };
  }

  _executeCommandTapeDeferred(tape: WebGPUCommandTape, inputSlots: Slot[], outputSlots: Slot[], outputTableIdxs: number[]) {`;

let startIndex = code.indexOf(searchStart);
if (startIndex === -1) throw new Error("Could not find executeCommandTape");

// Find closing brace
let braceCount = 0;
let inMethod = false;
let endIndex = -1;
for (let i = startIndex; i < code.length; i++) {
   if (code[i] === '{') {
     braceCount++;
     inMethod = true;
   }
   if (code[i] === '}') {
     braceCount--;
     if (inMethod && braceCount === 0) {
       endIndex = i;
       break;
     }
   }
}

let methodBody = code.substring(startIndex, endIndex);

methodBody = methodBody.replace('executeCommandTape(tape: WebGPUCommandTape, inputSlots: Slot[]): Slot[] {', '_executeCommandTapeDeferred(tape: WebGPUCommandTape, inputSlots: Slot[], outputSlots: Slot[], outputTableIdxs: number[]) {');

// Remove output slot creation from the deferred body
const endBlockReplace = `    // Create output slots
    const outputs: Slot[] = new globalThis.Array(tape.outputTableIdxs.length);
    for (let i = 0; i < tape.outputTableIdxs.length; i++) {
      const idx = tape.outputTableIdxs[i];
      const slot = this.nextSlot++;
      this.buffers.set(slot, {
        buffer: buffers[idx],
        size: sizes[idx],
        ref: 1,
      });
      outputs[i] = slot;
    }
    return outputs;`;

if (methodBody.includes(endBlockReplace)) {
    methodBody = methodBody.replace(endBlockReplace, `    // Populate pre-created output slots!
    for (let i = 0; i < outputTableIdxs.length; i++) {
      const idx = outputTableIdxs[i];
      this.buffers.get(outputSlots[i])!.buffer = buffers[idx];
    }`);
} else {
    console.log("Could not find end block, checking manually");
    // fallback replacing
    methodBody = methodBody.replace(/\/\/ Create output slots[\s\S]+return outputs;/, `    // Populate pre-created output slots!
    for (let i = 0; i < outputTableIdxs.length; i++) {
      const idx = outputTableIdxs[i];
      this.buffers.get(outputSlots[i])!.buffer = buffers[idx];
    }`);
}

const newCode = code.substring(0, startIndex) + replacementCode + '\n    ' + methodBody.substring(methodBody.indexOf('{') + 1) + code.substring(endIndex);

fs.writeFileSync('src/backend/webgpu.ts', newCode);
