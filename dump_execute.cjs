const fs = require('fs');
const lines = fs.readFileSync('src/backend/webgpu.ts', 'utf8').split('\n');
let inMethod = false;
let braceCount = 0;
let methodText = [];
for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('executeCommandTape(tape:')) {
        inMethod = true;
    }
    if (inMethod) {
        methodText.push(lines[i]);
        braceCount += (lines[i].match(/\{/g) || []).length;
        braceCount -= (lines[i].match(/\}/g) || []).length;
        if (braceCount === 0 && methodText.length > 2) {
            break;
        }
    }
}
fs.writeFileSync('tmp/exec.txt', methodText.join('\n'));
