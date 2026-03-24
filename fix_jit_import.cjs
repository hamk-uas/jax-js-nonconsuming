const fs = require('fs');
let code = fs.readFileSync('src/frontend/jit.ts', 'utf8');

code = code.replace(/PendingExecute/g, 'IPendingExecute');

fs.writeFileSync('src/frontend/jit.ts', code);
