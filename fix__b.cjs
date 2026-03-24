const fs = require('fs');

let jit = fs.readFileSync('src/frontend/jit.ts', 'utf8');
jit = jit.replace(/const outputSlots = \(this\.backend as WebGPUBackend\)\.executeCommandTape\(\s*this\._commandTape,\s*inputs,\s*\);\s*return \{ outputs: outputSlots, pending: \[\] \};/g, 'return (this.backend as WebGPUBackend).executeCommandTape(this._commandTape, inputs);');
fs.writeFileSync('src/frontend/jit.ts', jit);

let array = fs.readFileSync('src/frontend/array.ts', 'utf8');
array = array.replace(/export interface IPendingExecute \{/, 'export interface IPendingExecute {\n  readonly backend: Backend;');
fs.writeFileSync('src/frontend/array.ts', array);
