/**
 * @file Peephole optimizer for wasmblr instruction streams.
 *
 * Operates on raw WASM function body bytecodes after emission and before
 * binary encoding in `CodeGenerator.finish()`. Parses the flat byte stream
 * into structured instructions, applies safe local rewrites, and re-encodes.
 *
 * Design constraints:
 * - Zero external dependencies (self-contained in the wasmblr pipeline).
 * - Only applies semantics-preserving, block-local rewrites.
 * - Handles every opcode that wasmblr can emit; unknown opcodes abort the
 *   pass and return the original bytes unchanged (fail-safe).
 *
 * Current rewrite rules:
 *   1. local.set X ; local.get X  →  local.tee X        (fuse set+get)
 *   2. i32.const IDENTITY ; op   →  (remove both)       (identity element)
 *      Covers: add(0), sub(0), mul(1), and(-1), or(0), xor(0),
 *              shl(0), shr_s(0), shr_u(0)
 *   3. i32.const N ; i32.mul     →  i32.const log₂N ; i32.shl
 *                                     (strength reduction, N = power of 2)
 *   4. local.tee X ; drop        →  local.set X         (canonicalize)
 *   5. local.set X ; local.set X →  drop ; local.set X  (dead set)
 *   6. i32.const A ; i32.const B ; i32.binop → i32.const (A op B)
 *      (constant folding for add, sub, mul, and, or, xor, shl, shr_s, shr_u)
 *   7. i32.const N ; i32.add ; load/store offset=M → load/store offset=M+N
 *      (offset absorption, inspired by Binaryen OptimizeAddedConstants, N ≥ 0)
 *   8. Leading i32.const 0 ; local.set X → remove
 *      (WASM locals default to 0 at function entry — zero-init is redundant)
 *   9. i32.const 1 ; i32.lt_u  → i32.eqz   (comparison simplification)
 *      i32.const 0 ; i32.eq   → i32.eqz   (x <u 1 ≡ x == 0 ≡ eqz(x))
 */

// ---------------------------------------------------------------------------
// LEB128 helpers (duplicated from wasmblr.ts to keep module self-contained)
// ---------------------------------------------------------------------------

function decodeSigned(bytes: number[], pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  let byte: number;
  do {
    byte = bytes[pos++];
    result |= (byte & 0x7f) << shift;
    shift += 7;
  } while (byte & 0x80);
  // Sign-extend if the sign bit of the last byte is set.
  if (shift < 32 && byte & 0x40) result |= -(1 << shift);
  return [result, pos];
}

function decodeUnsigned(bytes: number[], pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  let byte: number;
  do {
    byte = bytes[pos++];
    result |= (byte & 0x7f) << shift;
    shift += 7;
  } while (byte & 0x80);
  return [result >>> 0, pos]; // coerce to uint32
}

function encodeSigned(n: number): number[] {
  const out: number[] = [];
  let more = true;
  while (more) {
    let byte = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (byte & 0x40) === 0) || (n === -1 && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    out.push(byte);
  }
  return out;
}

function encodeUnsigned(n: number): number[] {
  const out: number[] = [];
  do {
    let byte = n & 0x7f;
    n = n >>> 7;
    if (n !== 0) byte |= 0x80;
    out.push(byte);
  } while (n !== 0);
  return out;
}

// ---------------------------------------------------------------------------
// Instruction representation
// ---------------------------------------------------------------------------

const enum InstrKind {
  /** Instruction we understand and can transform. */
  Known,
  /** Opaque blob we cannot (or need not) transform. */
  Opaque,
  /** Marker: instruction was deleted by a rewrite rule. */
  Deleted,
}

interface KnownInstr {
  kind: InstrKind.Known;
  /** Primary opcode byte. */
  op: number;
  /** For local.get/set/tee (0x20-0x22): the local index. */
  localIdx?: number;
  /** For i32.const (0x41): the constant value. */
  i32Value?: number;
  /** For load/store memarg: log₂ alignment. */
  memAlign?: number;
  /** For load/store memarg: byte offset. */
  memOffset?: number;
}

interface OpaqueInstr {
  kind: InstrKind.Opaque;
  /** Raw bytes — copied through unchanged. */
  raw: number[];
}

interface DeletedInstr {
  kind: InstrKind.Deleted;
}

type Instr = KnownInstr | OpaqueInstr | DeletedInstr;

// Simple (no-immediate) opcodes that wasmblr can emit.
// All of these are single-byte opcodes with no trailing immediates.
const SIMPLE_OPCODES = new Set<number>([
  // control
  0x00, // unreachable
  0x01, // nop
  0x05, // else
  0x0b, // end
  0x0f, // return
  0x1a, // drop
  0x1b, // select
  // i32 arithmetic / comparison (0x45 – 0x78)
  0x45,
  0x46,
  0x47,
  0x48,
  0x49,
  0x4a,
  0x4b,
  0x4c,
  0x4d,
  0x4e,
  0x4f,
  0x67,
  0x68,
  0x69,
  0x6a,
  0x6b,
  0x6c,
  0x6d,
  0x6e,
  0x6f,
  0x70,
  0x71,
  0x72,
  0x73,
  0x74,
  0x75,
  0x76,
  0x77,
  0x78,
  // i64 comparison ops (0x50 – 0x5a)
  0x50,
  0x51,
  0x52,
  0x53,
  0x54,
  0x55,
  0x56,
  0x57,
  0x58,
  0x59,
  0x5a,
  // i64 arithmetic ops (0x79 – 0x8a)
  0x79,
  0x7a,
  0x7b,
  0x7c,
  0x7d,
  0x7e,
  0x7f,
  0x80,
  0x81,
  0x82,
  0x83,
  0x84,
  0x85,
  0x86,
  0x87,
  0x88,
  0x89,
  0x8a,
  // f32 operations (0x5b – 0x60, 0x8b – 0x98)
  0x5b,
  0x5c,
  0x5d,
  0x5e,
  0x5f,
  0x60,
  0x8b,
  0x8c,
  0x8d,
  0x8e,
  0x8f,
  0x90,
  0x91,
  0x92,
  0x93,
  0x94,
  0x95,
  0x96,
  0x97,
  0x98,
  // f64 operations (0x61 – 0x66, 0x99 – 0xa6)
  0x61,
  0x62,
  0x63,
  0x64,
  0x65,
  0x66,
  0x99,
  0x9a,
  0x9b,
  0x9c,
  0x9d,
  0x9e,
  0x9f,
  0xa0,
  0xa1,
  0xa2,
  0xa3,
  0xa4,
  0xa5,
  0xa6,
  // conversions — ALL single-byte conversion ops (0xa7 – 0xbf)
  0xa7, // i32.wrap_i64
  0xa8, // i32.trunc_f32_s
  0xa9, // i32.trunc_f32_u
  0xaa, // i32.trunc_f64_s
  0xab, // i32.trunc_f64_u
  0xac, // i64.extend_i32_s
  0xad, // i64.extend_i32_u
  0xae, // i64.trunc_f32_s
  0xaf, // i64.trunc_f32_u
  0xb0, // i64.trunc_f64_s
  0xb1, // i64.trunc_f64_u
  0xb2, // f32.convert_i32_s
  0xb3, // f32.convert_i32_u
  0xb4, // f32.convert_i64_s
  0xb5, // f32.convert_i64_u
  0xb6, // f32.demote_f64
  0xb7, // f64.convert_i32_s
  0xb8, // f64.convert_i32_u
  0xb9, // f64.convert_i64_s
  0xba, // f64.convert_i64_u
  0xbb, // f64.promote_f32
  0xbc, // i32.reinterpret_f32
  0xbd, // i64.reinterpret_f64
  0xbe, // f32.reinterpret_i32
  0xbf, // f64.reinterpret_i64
]);

// Opcodes that take a single unsigned LEB128 immediate.
const _U32_IMM_OPCODES = new Set<number>([
  0x0c, // br
  0x0d, // br_if
  0x10, // call
  0x20, // local.get
  0x21, // local.set
  0x22, // local.tee
]);

// Memory load/store opcodes: 2 × unsigned LEB128 (align + offset).
const MEMARG_OPCODES = new Set<number>([
  0x28,
  0x29,
  0x2a,
  0x2b,
  0x2c,
  0x2d,
  0x2e,
  0x2f, // loads
  0x36,
  0x37,
  0x38,
  0x39,
  0x3a,
  0x3b,
  0x3c,
  0x3d,
  0x3e, // stores
]);

// Block opcodes: blocktype immediate (signed LEB128).
const BLOCK_OPCODES = new Set<number>([
  0x02, // block
  0x03, // loop
  0x04, // if
]);

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

function parseInstructions(bytes: number[]): Instr[] | null {
  const instrs: Instr[] = [];
  let pos = 0;

  while (pos < bytes.length) {
    const startPos = pos;
    const op = bytes[pos++];

    // -- Known instructions we may transform ---
    if (op >= 0x20 && op <= 0x22) {
      // local.get / local.set / local.tee
      const [idx, next] = decodeUnsigned(bytes, pos);
      pos = next;
      instrs.push({ kind: InstrKind.Known, op, localIdx: idx });
      continue;
    }

    if (op === 0x41) {
      // i32.const
      const [val, next] = decodeSigned(bytes, pos);
      pos = next;
      instrs.push({ kind: InstrKind.Known, op, i32Value: val });
      continue;
    }

    // -- Simple no-immediate opcodes ---
    if (SIMPLE_OPCODES.has(op)) {
      instrs.push({ kind: InstrKind.Known, op });
      continue;
    }

    // -- Opcodes with a single unsigned LEB128 immediate (opaque) ---
    if (op === 0x0c || op === 0x0d || op === 0x10) {
      // br / br_if / call — skip label depth or function index
      const [, next] = decodeUnsigned(bytes, pos);
      pos = next;
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0x42) {
      // i64.const — signed LEB128
      const [, next] = decodeSigned(bytes, pos);
      pos = next;
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    // -- Opcodes that we pass through as opaque blobs ---

    if (MEMARG_OPCODES.has(op)) {
      // 2 unsigned LEB128 immediates (align + offset)
      const [align, next1] = decodeUnsigned(bytes, pos);
      const [offset, next2] = decodeUnsigned(bytes, next1);
      pos = next2;
      instrs.push({
        kind: InstrKind.Known,
        op,
        memAlign: align,
        memOffset: offset,
      });
      continue;
    }

    if (BLOCK_OPCODES.has(op)) {
      // blocktype: signed LEB128
      const [, next] = decodeSigned(bytes, pos);
      pos = next;
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0x0e) {
      // br_table: u32 count, then (count+1) u32 labels
      const [count, next] = decodeUnsigned(bytes, pos);
      pos = next;
      for (let i = 0; i <= count; i++) {
        const [, next2] = decodeUnsigned(bytes, pos);
        pos = next2;
      }
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0x43) {
      // f32.const: 4 bytes
      pos += 4;
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0x44) {
      // f64.const: 8 bytes
      pos += 8;
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0x3f || op === 0x40) {
      // memory.size / memory.grow: 1 byte (0x00)
      pos++;
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0xfc) {
      // Multi-byte prefix: trunc_sat / bulk memory
      const [subOp, next] = decodeUnsigned(bytes, pos);
      pos = next;
      if (subOp === 0x0a) {
        // memory.copy: 2 zero bytes
        pos += 2;
      } else if (subOp === 0x0b) {
        // memory.fill: 1 zero byte
        pos += 1;
      }
      // trunc_sat (0x00-0x07): no extra immediates
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    if (op === 0xfd) {
      // SIMD prefix
      const [subOp, next] = decodeUnsigned(bytes, pos);
      pos = next;
      if (subOp <= 0x0b) {
        // v128.load / v128.store variants: align + offset
        const [, next1] = decodeUnsigned(bytes, pos);
        const [, next2] = decodeUnsigned(bytes, next1);
        pos = next2;
      } else if (subOp === 0x0c) {
        // v128.const: 16 bytes immediate
        pos += 16;
      } else if (subOp === 0x0d) {
        // i8x16.shuffle: 16 bytes immediate (lane indices)
        pos += 16;
      } else if (
        // Lane extract/replace ops: 1-byte lane immediate
        subOp >= 0x15 &&
        subOp <= 0x22 // i8x16..f64x2 extract/replace
      ) {
        pos++; // lane byte
      }
      // All other SIMD ops: no extra immediates
      instrs.push({ kind: InstrKind.Opaque, raw: bytes.slice(startPos, pos) });
      continue;
    }

    // Unknown opcode — bail out and return null (fail-safe).
    return null;
  }

  return instrs;
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

function encodeInstruction(instr: Instr): number[] {
  if (instr.kind === InstrKind.Deleted) return [];
  if (instr.kind === InstrKind.Opaque) return instr.raw;

  const { op } = instr;

  // local.get / local.set / local.tee
  if (op >= 0x20 && op <= 0x22) {
    return [op, ...encodeUnsigned(instr.localIdx!)];
  }
  // i32.const
  if (op === 0x41) {
    return [op, ...encodeSigned(instr.i32Value!)];
  }
  // load/store with memarg
  if (instr.memAlign !== undefined) {
    return [
      op,
      ...encodeUnsigned(instr.memAlign),
      ...encodeUnsigned(instr.memOffset!),
    ];
  }
  // All other known opcodes are single-byte, no immediates.
  return [op];
}

// ---------------------------------------------------------------------------
// Peephole rewriter
// ---------------------------------------------------------------------------

export interface PeepholeStats {
  /** local.set X ; local.get X → local.tee X */
  setGetToTee: number;
  /** i32.const IDENTITY ; op → removed (add(0), sub(0), mul(1), etc.) */
  identity: number;
  /** i32.const N ; i32.mul → i32.const log2(N) ; i32.shl */
  strengthReduction: number;
  /** local.tee X ; drop → local.set X */
  teeDropToSet: number;
  /** local.set X ; local.set X → drop ; local.set X */
  deadSet: number;
  /** i32.const A ; i32.const B ; binop → i32.const (A op B) */
  constFold: number;
  /** i32.const N ; i32.add ; load/store offset=M → load/store offset=M+N */
  offsetAbsorb: number;
  /** Leading i32.const 0 ; local.set X removed (WASM locals default to 0) */
  zeroInit: number;
  /** i32.const 1 ; i32.lt_u → i32.eqz (and i32.const 0 ; i32.eq → i32.eqz) */
  compSimplify: number;
  /** Total instructions before optimization. */
  instrsBefore: number;
  /** Total instructions after optimization. */
  instrsAfter: number;
}

function emptyStats(): PeepholeStats {
  return {
    setGetToTee: 0,
    identity: 0,
    strengthReduction: 0,
    teeDropToSet: 0,
    deadSet: 0,
    constFold: 0,
    offsetAbsorb: 0,
    zeroInit: 0,
    compSimplify: 0,
    instrsBefore: 0,
    instrsAfter: 0,
  };
}

function isPowerOf2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function log2int(n: number): number {
  let r = 0;
  while (n > 1) {
    n >>>= 1;
    r++;
  }
  return r;
}

// Identity element for each i32 binary operation.
// When i32.const V precedes the operation and V equals the identity, both
// instructions can be deleted (x ⊕ identity = x).
const IDENTITY_MAP = new Map<number, number>([
  [0x6a, 0], // i32.add:   x + 0 = x
  [0x6b, 0], // i32.sub:   x - 0 = x
  [0x6c, 1], // i32.mul:   x * 1 = x
  [0x71, -1], // i32.and:  x & -1 = x
  [0x72, 0], // i32.or:    x | 0 = x
  [0x73, 0], // i32.xor:   x ^ 0 = x
  [0x74, 0], // i32.shl:   x << 0 = x
  [0x75, 0], // i32.shr_s: x >> 0 = x
  [0x76, 0], // i32.shr_u: x >>> 0 = x
]);

// Constant-folding evaluators for i32 binary operations.
const I32_FOLD = new Map<number, (a: number, b: number) => number>([
  [0x6a, (a, b) => (a + b) | 0],
  [0x6b, (a, b) => (a - b) | 0],
  [0x6c, (a, b) => Math.imul(a, b)],
  [0x71, (a, b) => a & b],
  [0x72, (a, b) => a | b],
  [0x73, (a, b) => a ^ b],
  [0x74, (a, b) => (a << (b & 31)) | 0],
  [0x75, (a, b) => a >> (b & 31)],
  [0x76, (a, b) => (a >>> (b & 31)) | 0],
]);

function isKnown(instr: Instr): instr is KnownInstr {
  return instr.kind === InstrKind.Known;
}

/**
 * Run peephole rewrites on the instruction stream (in-place, multi-pass).
 * Returns combined stats from all passes. Runs until no more rewrites fire.
 */
function peepholeRewrite(instrs: Instr[], stats: PeepholeStats): void {
  // Rule 8: Remove leading i32.const 0 ; local.set X pairs at function start.
  // WASM locals are zero-initialized by spec, so these are redundant.
  // Only consecutive pairs from the start are removed (no dataflow analysis).
  for (let i = 0; i + 1 < instrs.length; i += 2) {
    const a = instrs[i];
    const b = instrs[i + 1];
    if (
      a.kind === InstrKind.Known &&
      a.op === 0x41 &&
      a.i32Value === 0 &&
      b.kind === InstrKind.Known &&
      b.op === 0x21
    ) {
      instrs[i] = { kind: InstrKind.Deleted };
      instrs[i + 1] = { kind: InstrKind.Deleted };
      stats.zeroInit++;
    } else {
      break; // Stop at first non-zero-init instruction
    }
  }

  let changed = true;
  while (changed) {
    changed = false;

    // Compact deleted instructions so adjacent patterns become visible
    // across passes (enables cascading, e.g. strength reduction → const fold).
    let w = 0;
    for (let r = 0; r < instrs.length; r++) {
      if (instrs[r].kind !== InstrKind.Deleted) instrs[w++] = instrs[r];
    }
    instrs.length = w;

    for (let i = 0; i < instrs.length - 1; i++) {
      const a = instrs[i];
      const b = instrs[i + 1];
      if (!isKnown(a) || !isKnown(b)) continue;

      // Rule 1: local.set X ; local.get X → local.tee X
      if (a.op === 0x21 && b.op === 0x20 && a.localIdx === b.localIdx) {
        instrs[i] = { kind: InstrKind.Known, op: 0x22, localIdx: a.localIdx };
        instrs[i + 1] = { kind: InstrKind.Deleted };
        stats.setGetToTee++;
        changed = true;
        continue;
      }

      // Rule 2: i32.const IDENTITY ; op → delete both (table-driven)
      if (a.op === 0x41) {
        const identityVal = IDENTITY_MAP.get(b.op);
        if (identityVal !== undefined && a.i32Value === identityVal) {
          instrs[i] = { kind: InstrKind.Deleted };
          instrs[i + 1] = { kind: InstrKind.Deleted };
          stats.identity++;
          changed = true;
          continue;
        }
      }

      // Rule 3: i32.const N ; i32.mul → i32.const log2(N) ; i32.shl
      //         where N is a power of 2 and N > 1
      if (
        a.op === 0x41 &&
        b.op === 0x6c &&
        a.i32Value! > 1 &&
        isPowerOf2(a.i32Value!)
      ) {
        instrs[i] = {
          kind: InstrKind.Known,
          op: 0x41,
          i32Value: log2int(a.i32Value!),
        };
        instrs[i + 1] = { kind: InstrKind.Known, op: 0x74 }; // i32.shl
        stats.strengthReduction++;
        changed = true;
        continue;
      }

      // Rule 4: local.tee X ; drop → local.set X
      if (a.op === 0x22 && b.op === 0x1a) {
        instrs[i] = { kind: InstrKind.Known, op: 0x21, localIdx: a.localIdx };
        instrs[i + 1] = { kind: InstrKind.Deleted };
        stats.teeDropToSet++;
        changed = true;
        continue;
      }

      // Rule 9: i32.const 1 ; i32.lt_u → i32.eqz  (x <u 1 ≡ x == 0)
      //          i32.const 0 ; i32.eq   → i32.eqz  (eq(x,0) ≡ eqz(x))
      if (
        (a.op === 0x41 && a.i32Value === 1 && b.op === 0x49) || // const 1 ; lt_u
        (a.op === 0x41 && a.i32Value === 0 && b.op === 0x46) // const 0 ; eq
      ) {
        instrs[i] = { kind: InstrKind.Deleted };
        instrs[i + 1] = { kind: InstrKind.Known, op: 0x45 }; // i32.eqz
        stats.compSimplify++;
        changed = true;
        continue;
      }

      // Rule 5: local.set X ; local.set X → drop ; local.set X
      if (a.op === 0x21 && b.op === 0x21 && a.localIdx === b.localIdx) {
        instrs[i] = { kind: InstrKind.Known, op: 0x1a }; // drop
        stats.deadSet++;
        changed = true;
        continue;
      }

      // --- 3-instruction window rules ---
      if (i >= instrs.length - 2) continue;
      const c = instrs[i + 2];
      if (!isKnown(c)) continue;

      // Rule 6: i32.const A ; i32.const B ; i32.binop → i32.const (A op B)
      //         (constant folding — inspired by Binaryen Precompute)
      if (a.op === 0x41 && b.op === 0x41) {
        const fn = I32_FOLD.get(c.op);
        if (fn) {
          instrs[i] = {
            kind: InstrKind.Known,
            op: 0x41,
            i32Value: fn(a.i32Value!, b.i32Value!),
          };
          instrs[i + 1] = { kind: InstrKind.Deleted };
          instrs[i + 2] = { kind: InstrKind.Deleted };
          stats.constFold++;
          changed = true;
          continue;
        }
      }

      // Rule 7: i32.const N ; i32.add ; load/store offset=M → offset=M+N
      //         (inspired by Binaryen OptimizeAddedConstants)
      if (
        a.op === 0x41 &&
        a.i32Value! >= 0 &&
        b.op === 0x6a && // i32.add
        c.memOffset !== undefined
      ) {
        const newOffset = a.i32Value! + c.memOffset!;
        if (newOffset <= 0xffffffff) {
          instrs[i] = { kind: InstrKind.Deleted };
          instrs[i + 1] = { kind: InstrKind.Deleted };
          (c as KnownInstr).memOffset = newOffset;
          stats.offsetAbsorb++;
          changed = true;
          continue;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Apply peephole optimization to a WASM function body byte stream.
 *
 * Returns the optimized bytes, or the original bytes unchanged if parsing
 * fails (fail-safe). If `stats` is provided, optimization counts are
 * accumulated into it.
 */
export function optimizeFunctionBody(
  bytes: number[],
  stats?: PeepholeStats,
): number[] {
  const instrs = parseInstructions(bytes);
  if (instrs === null) return bytes; // parse failed — return unchanged

  const localStats = stats ?? emptyStats();
  localStats.instrsBefore += instrs.length;

  peepholeRewrite(instrs, localStats);

  // Re-encode
  const out: number[] = [];
  let count = 0;
  for (const instr of instrs) {
    if (instr.kind !== InstrKind.Deleted) {
      out.push(...encodeInstruction(instr));
      count++;
    }
  }
  localStats.instrsAfter += count;

  return out;
}

/**
 * Format peephole stats as a human-readable summary string.
 */
export function formatPeepholeStats(stats: PeepholeStats): string {
  const lines: string[] = [];
  const total =
    stats.setGetToTee +
    stats.identity +
    stats.strengthReduction +
    stats.teeDropToSet +
    stats.deadSet +
    stats.constFold +
    stats.offsetAbsorb +
    stats.zeroInit +
    stats.compSimplify;

  if (total === 0) return "wasmblr peephole: no rewrites applied";

  lines.push(
    `wasmblr peephole: ${total} rewrite(s), ${stats.instrsBefore} → ${stats.instrsAfter} instructions`,
  );
  if (stats.setGetToTee) lines.push(`  set+get→tee: ${stats.setGetToTee}`);
  if (stats.identity) lines.push(`  identity removal: ${stats.identity}`);
  if (stats.strengthReduction)
    lines.push(`  strength reduction (mul→shl): ${stats.strengthReduction}`);
  if (stats.teeDropToSet) lines.push(`  tee+drop→set: ${stats.teeDropToSet}`);
  if (stats.deadSet) lines.push(`  dead set: ${stats.deadSet}`);
  if (stats.constFold) lines.push(`  constant fold: ${stats.constFold}`);
  if (stats.offsetAbsorb)
    lines.push(`  offset absorption: ${stats.offsetAbsorb}`);
  if (stats.zeroInit) lines.push(`  zero-init removal: ${stats.zeroInit}`);
  if (stats.compSimplify)
    lines.push(`  comparison simplification: ${stats.compSimplify}`);

  return lines.join("\n");
}

export function newPeepholeStats(): PeepholeStats {
  return emptyStats();
}
