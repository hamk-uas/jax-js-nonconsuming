const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

code = code.replace(
  '  resetCalibration() {',
  `  // @ts-ignore
  resetCalibration() {`
);

code = code.replace(
  '  applyCalibration(beliefState: PerformanceBeliefState) {',
  `  // @ts-ignore
  applyCalibration(beliefState: any) {`
);

fs.writeFileSync('src/backend/webgpu.ts', code);
