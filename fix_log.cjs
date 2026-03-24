const fs = require('fs');

let webgpu = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

webgpu = webgpu.replace(/              \/\/ Dispatch scatter_add kernel/, `              // Dispatch scatter_add kernel\n              if (!buffers[sa.indicesIdx]) { console.error("indicesIdx buf undefined! ", sa.indicesIdx); }\n              if (!buffers[sa.targetIdx]) { console.error("targetIdx buf undefined! ", sa.targetIdx); }\n              if (!buffers[sa.updatesIdx]) { console.error("updatesIdx buf undefined! ", sa.updatesIdx); }`);

fs.writeFileSync('src/backend/webgpu.ts', webgpu);
