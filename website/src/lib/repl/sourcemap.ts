/**
 * Pure source-map remapping helpers for the REPL.
 *
 * These take a Rollup-style source map and convert generated (bundled) code
 * positions back to the user's original TypeScript source positions.
 */

export interface LeakMarker {
  line: number;
  message: string;
}

export type SourceMapLike = {
  mappings?: string;
};

const BASE64 =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

export function decodeVlq(mappings: string, start: number): [number, number] {
  let result = 0;
  let shift = 0;
  let index = start;
  let continuation = false;

  do {
    if (index >= mappings.length) {
      return [0, index];
    }
    const digit = BASE64.indexOf(mappings[index++]);
    if (digit < 0) {
      return [0, index];
    }
    continuation = (digit & 32) !== 0;
    const value = digit & 31;
    result += value << shift;
    shift += 5;
  } while (continuation);

  const isNegative = (result & 1) === 1;
  result >>= 1;
  return [isNegative ? -result : result, index];
}

/**
 * Find the original source position for a generated line/column.
 *
 * Both `generatedLine` and `generatedColumn` are **1-based** (matching the
 * format used by V8 stack traces: `index.ts:LINE:COL`).
 */
export function mapGeneratedPositionToSource(
  map: SourceMapLike | null,
  generatedLine: number,
  generatedColumn: number,
): { line: number; column: number } | null {
  const mappings = map?.mappings;
  if (!mappings || generatedLine <= 0 || generatedColumn <= 0) return null;

  let index = 0;
  let line = 1;
  let source = 0;
  let sourceLine = 0;
  let sourceColumn = 0;
  let name = 0;

  while (index <= mappings.length) {
    let generatedColumnState = 0;
    let best: { line: number; column: number } | null = null;

    while (index < mappings.length) {
      const ch = mappings[index];
      if (ch === ";") {
        index++;
        break;
      }
      if (ch === ",") {
        index++;
        continue;
      }

      let delta = 0;
      [delta, index] = decodeVlq(mappings, index);
      generatedColumnState += delta;

      let hasSource = false;
      if (
        index < mappings.length &&
        mappings[index] !== "," &&
        mappings[index] !== ";"
      ) {
        hasSource = true;
        [delta, index] = decodeVlq(mappings, index);
        source += delta;
        [delta, index] = decodeVlq(mappings, index);
        sourceLine += delta;
        [delta, index] = decodeVlq(mappings, index);
        sourceColumn += delta;

        if (
          index < mappings.length &&
          mappings[index] !== "," &&
          mappings[index] !== ";"
        ) {
          [delta, index] = decodeVlq(mappings, index);
          name += delta;
        }
      }

      if (
        line === generatedLine &&
        hasSource &&
        generatedColumnState <= generatedColumn - 1
      ) {
        best = {
          line: sourceLine + 1,
          column: sourceColumn + 1,
        };
      }
    }

    if (line === generatedLine) {
      return best;
    }

    if (index >= mappings.length) break;
    line++;
  }

  return null;
}

/**
 * Replace `index.ts:LINE:COL` references in text with source-mapped positions.
 */
export function remapReplLocationText(
  text: string,
  map: SourceMapLike | null,
): string {
  return text.replace(
    /((?:.*\/)?(?:index|main)\.ts):(\d+):(\d+)/g,
    (_all, filePath: string, lineRaw: string, colRaw: string) => {
      const line = parseInt(lineRaw, 10);
      const col = parseInt(colRaw, 10);
      const mapped = mapGeneratedPositionToSource(map, line, col);
      if (!mapped) return `${filePath}:${line}:${col}`;
      return `${filePath}:${mapped.line}:${mapped.column}`;
    },
  );
}

export function remapLeakDetails(
  reportDetails: string[],
  map: SourceMapLike | null,
): string[] {
  return reportDetails.map((detail) => remapReplLocationText(detail, map));
}

/**
 * Extract leak markers (line number + message) from remapped detail strings.
 *
 * Input format: `"Array:float32[] created at index.ts:3:7"`
 * Output: `{ line: 3, message: "Leaked: Array:float32[]. Use `using` or call .dispose()" }`
 */
export function parseLeakMarkers(reportDetails: string[]): LeakMarker[] {
  const markers: LeakMarker[] = [];
  for (const detail of reportDetails) {
    const m = detail.match(
      /^(.+?) created at (?:(?:.*\/)?(?:index|main)\.ts):(\d+):(\d+)$/,
    );
    if (m) {
      markers.push({
        line: parseInt(m[2]),
        message: `Leaked: ${m[1]}. Use \`using\` or call .dispose()`,
      });
    }
  }
  return markers;
}
