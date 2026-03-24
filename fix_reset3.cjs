const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

code = code.replace(`    // @ts-ignore
    resetCalibration() {
      // @ts-ignore
      this.capabilities.calibrated = false;
    }
      this.capabilities.calibrated = false;
    }`, `    // @ts-ignore
    resetCalibration() {
      // @ts-ignore
      this.capabilities.calibrated = false;
    }`);

fs.writeFileSync('src/backend/webgpu.ts', code);
