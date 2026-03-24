const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');
code = code.replace(/import type \{ JitId, JitStep \} from "\.\.\/frontend\/jit";/g, 'import type { IPendingExecute } from "../frontend/array.js";\nimport type { JitId, JitStep } from "../frontend/jit";');
fs.writeFileSync('src/backend/webgpu.ts', code);

for (const file of ['src/frontend/block-map-executor.ts', 'src/frontend/scan-executor.ts', 'src/frontend/jit.ts']) {
    let content = fs.readFileSync(file, 'utf8');
    if (!content.includes('IPendingExecute')) {
        content = content.replace(/import \{.*?PendingExecute.*?\} from "\.\/array\.js"/g, 'import { PendingExecute, type IPendingExecute } from "./array.js"');
    }
    fs.writeFileSync(file, content);
}
