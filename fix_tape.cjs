const fs = require('fs');

let webgpu = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

// Ensure IPendingExecute import
if (!webgpu.includes('IPendingExecute')) {
    webgpu = webgpu.replace('import type { Backend, ExecutionContext } from "../backend.js";', 'import type { Backend, ExecutionContext } from "../backend.js";\nimport type { IPendingExecute } from "../frontend/array.js";');
}

fs.writeFileSync('src/backend/webgpu.ts', webgpu);
