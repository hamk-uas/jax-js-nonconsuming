const fs = require('fs');
let code = fs.readFileSync('src/frontend/array.ts', 'utf8');

code = code.replace(
  'export class PendingExecute {',
  `export interface IPendingExecute {
  prepared: any;
  submitted: boolean;
  updateRc(delta: number): void;
  prepare(): Promise<void>;
  prepareSync(): void;
  submit(): void;
}

export class PendingExecute implements IPendingExecute {`
);

// replace types of pending
code = code.replace(/PendingExecute\[\]/g, 'IPendingExecute[]');
code = code.replace(/Iterable<PendingExecute>/g, 'Iterable<IPendingExecute>');
code = code.replace(/Set<PendingExecute>/g, 'Set<IPendingExecute>');

fs.writeFileSync('src/frontend/array.ts', code);
