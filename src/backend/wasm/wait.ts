export function canWaitOnThisThread(): boolean {
  try {
    const probe = new Int32Array(new SharedArrayBuffer(4));
    Atomics.wait(probe, 0, 1, 0);
    return true;
  } catch {
    return false;
  }
}

export function waitWhileState(
  control: Int32Array,
  index: number,
  expected: number,
  canWait: boolean,
): void {
  if (canWait) {
    Atomics.wait(control, index, expected);
    return;
  }
  while (Atomics.load(control, index) === expected) {
    // busy wait fallback
  }
}
