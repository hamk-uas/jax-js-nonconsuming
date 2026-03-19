#!/usr/bin/env node
/**
 * refit-model.mjs — Incrementally update the GPU cost model
 *
 * The model stores Welford mean-centered sufficient statistics alongside
 * the derived regression coefficients. This script:
 *   1. Adds new GPU data points to the sufficient stats (Welford update)
 *   2. Re-derives all regression coefficients, medians, and Apple lookup
 *   3. Writes the updated runtime_model.json back
 *
 * Usage:
 *   # Add a single GPU:
 *   node scripts/refit-model.mjs add \
 *     --model "NVIDIA_RTX_5090" \
 *     --vendorClass "discrete-modern" \
 *     --maxBufGB 32 \
 *     --bwGBs 1792 \
 *     --tflops 105
 *
 *   # Add an Apple GPU (include --appleGen for lookup table):
 *   node scripts/refit-model.mjs add \
 *     --model "Apple_M5_12C" \
 *     --vendorClass "apple" \
 *     --maxBufGB 16 \
 *     --bwGBs 200 \
 *     --tflops 8.5 \
 *     --appleGen "common-5"
 *
 *   # Just re-derive coefficients from existing stats (after manual edit):
 *   node scripts/refit-model.mjs refit
 *
 * Data sources for new GPUs:
 *   - Bandwidth / TFLOPS: TechPowerUp GPU specs database
 *   - maxBufferSize: Dawn source defaults or local WebGPU adapter query
 *   - Features: WebGPU CTS conformance reports or local adapter.features
 */

// ── OLS from Welford stats ──
function ols(n, w) {
  if (n < 2 || w.ssXX === 0) return { a: w.mean_y, b: 0, r2: 0 };
  const b = w.ssXY / w.ssXX;
  const a = w.mean_y - b * w.mean_x;
  const r2 = w.ssYY > 0 ? (w.ssXY * w.ssXY) / (w.ssXX * w.ssYY) : 0;
  return { a: r6(a), b: r6(b), r2: r6(r2) };
}

function r6(v) {
  return Math.round(v * 1e6) / 1e6;
}
function r8(v) {
  return parseFloat(v.toPrecision(8));
}

// ── Welford incremental update ──
// Given existing { mean_x, mean_y, ssXX, ssXY, ssYY } for n points,
// add one new (x, y) observation.
function welfordAdd(n, w, x, y) {
  const n1 = n + 1;
  const dx = x - w.mean_x;
  const dy = y - w.mean_y;
  const mx = w.mean_x + dx / n1;
  const my = w.mean_y + dy / n1;
  // Key insight: ssXX_new = ssXX_old + dx * (x - mx_new)
  return {
    mean_x: r8(mx),
    mean_y: r8(my),
    ssXX: r8(w.ssXX + dx * (x - mx)),
    ssXY: r8(w.ssXY + dx * (y - my)),
    ssYY: r8(w.ssYY + dy * (y - my)),
  };
}

// ── Update vendorModels from sufficient stats ──
function updateCoefficients(model) {
  const stats = model._sufficientStats;

  for (const [vc, ss] of Object.entries(stats.perClass)) {
    const vm = model.vendorModels[vc];
    if (!vm) continue;

    // Re-derive regression coefficients
    const bToBw = ols(ss.n, ss.bufToBw);
    const bwToTf = ols(ss.n, ss.bwToTf);
    const bToTf = ols(ss.n, ss.bufToTf);

    vm.bufToBw = { a: bToBw.a, b: bToBw.b, r2: bToBw.r2 };
    vm.bwToTf = { a: bwToTf.a, b: bwToTf.b, r2: bwToTf.r2 };
    vm.bufToTf = { a: bToTf.a, b: bToTf.b, r2: bToTf.r2 };
    vm.regressionUsable = ss.n >= 5 && bToBw.r2 >= 0.5;
  }

  // Global fallback
  const gs = stats.global;
  const gBToBw = ols(gs.n, gs.bufToBw);
  const gBwToTf = ols(gs.n, gs.bwToTf);
  const gBToTf = ols(gs.n, gs.bufToTf);
  model.globalFallback.bufToBw = { a: gBToBw.a, b: gBToBw.b, r2: gBToBw.r2 };
  model.globalFallback.bwToTf = { a: gBwToTf.a, b: gBwToTf.b, r2: gBwToTf.r2 };
  model.globalFallback.bufToTf = { a: gBToTf.a, b: gBToTf.b, r2: gBToTf.r2 };

  // Apple generation lookup — update geomeans from sufficient stats.
  // Only update entries that already exist in appleLookup (bw_range/tf_range/gpus
  // are metadata that require raw data points, populated by the `add` command).
  for (const [gen, s] of Object.entries(stats.appleGenerations)) {
    if (gen.startsWith("_")) continue;
    const bwGeo = Math.exp(s.sum_log_bw / s.n);
    const tfGeo = Math.exp(s.sum_log_tf / s.n);
    if (model.appleLookup[gen]) {
      const al = model.appleLookup[gen];
      al.n = s.n;
      al.bw_geomean = r6(bwGeo);
      al.tf_geomean = r6(tfGeo);
    }
  }

  // Note: _safari_fallback is NOT recomputed here. It was calibrated from
  // per-GPU P25/median across 14 individual Apple GPUs (not generation
  // geomeans). We only have generation aggregates in sufficient stats, which
  // would produce a non-conservative fallback. Manual edit if needed.

  // Update meta
  model._meta.vendorClassCounts = {};
  for (const [vc, ss] of Object.entries(stats.perClass)) {
    model._meta.vendorClassCounts[vc] = ss.n;
  }
  model._meta.trainingSet = `${gs.n} GPUs`;

  return model;
}

// ── Exported for testing ──
export { ols, r6, r8, welfordAdd, updateCoefficients, addGpu };

/**
 * Add a GPU data point to the model in-place.
 * @param {object} model - The runtime model object (mutated)
 * @param {{ name: string, vendorClass: string, maxBufBytes: number, bw: number, tf: number, appleGen?: string }} gpu
 */
function addGpu(
  model,
  { name, vendorClass: vc, maxBufBytes, bw, tf, appleGen },
) {
  if (vc === "apple" && !appleGen) {
    throw new Error(
      `appleGen is required when vendorClass is "apple" (GPU: ${name})`,
    );
  }

  const logBuf = Math.log(maxBufBytes);
  const logBw = Math.log(bw);
  const logTf = Math.log(tf);
  const stats = model._sufficientStats;

  // Update per-class stats
  if (!stats.perClass[vc]) {
    stats.perClass[vc] = {
      n: 0,
      gpus: [],
      bufToBw: { mean_x: logBuf, mean_y: logBw, ssXX: 0, ssXY: 0, ssYY: 0 },
      bwToTf: { mean_x: logBw, mean_y: logTf, ssXX: 0, ssXY: 0, ssYY: 0 },
      bufToTf: { mean_x: logBuf, mean_y: logTf, ssXX: 0, ssXY: 0, ssYY: 0 },
    };
  }
  const cs = stats.perClass[vc];

  // Lazy-init gpus list from existing model contents for seeded classes.
  // Apple names live in appleLookup[gen].gpus; non-Apple start empty.
  if (!cs.gpus || (cs.gpus.length === 0 && cs.n > 0)) {
    if (vc === "apple" && model.appleLookup) {
      cs.gpus = Object.entries(model.appleLookup)
        .filter(([k]) => !k.startsWith("_"))
        .flatMap(([, v]) => v.gpus ?? []);
    } else {
      if (cs.n > 0) {
        console.warn(
          `[duplicate-protection] Warning: class '${vc}' has ${cs.n} seeded data points without names. Duplicate detection cannot protect original seeded data.`,
        );
      }
      cs.gpus = [];
    }
  }

  // Guard: reject duplicate GPU names (Welford stats are not idempotent)
  if (cs.gpus.includes(name)) {
    return { n: cs.n, globalN: stats.global.n, duplicate: true };
  }
  cs.gpus.push(name);

  cs.bufToBw = welfordAdd(cs.n, cs.bufToBw, logBuf, logBw);
  cs.bwToTf = welfordAdd(cs.n, cs.bwToTf, logBw, logTf);
  cs.bufToTf = welfordAdd(cs.n, cs.bufToTf, logBuf, logTf);
  cs.n++;

  // Update global stats
  const gs = stats.global;
  gs.bufToBw = welfordAdd(gs.n, gs.bufToBw, logBuf, logBw);
  gs.bwToTf = welfordAdd(gs.n, gs.bwToTf, logBw, logTf);
  gs.bufToTf = welfordAdd(gs.n, gs.bufToTf, logBuf, logTf);
  gs.n++;

  // Apple generation — update both sufficient stats and lookup metadata
  if (vc === "apple" && appleGen) {
    const gen = appleGen;
    if (!stats.appleGenerations[gen])
      stats.appleGenerations[gen] = { n: 0, sum_log_bw: 0, sum_log_tf: 0 };
    stats.appleGenerations[gen].n++;
    stats.appleGenerations[gen].sum_log_bw = r8(
      stats.appleGenerations[gen].sum_log_bw + logBw,
    );
    stats.appleGenerations[gen].sum_log_tf = r8(
      stats.appleGenerations[gen].sum_log_tf + logTf,
    );

    // Ensure appleLookup entry exists with valid structure
    if (!model.appleLookup[gen]) {
      model.appleLookup[gen] = {
        n: 0,
        bw_range: [bw, bw],
        tf_range: [tf, tf],
        bw_geomean: 0,
        tf_geomean: 0,
        gpus: [],
      };
    }
    const al = model.appleLookup[gen];
    al.bw_range = [Math.min(al.bw_range[0], bw), Math.max(al.bw_range[1], bw)];
    al.tf_range = [Math.min(al.tf_range[0], tf), Math.max(al.tf_range[1], tf)];
    if (!al.gpus.includes(name)) al.gpus.push(name);
  }

  return { n: cs.n, globalN: gs.n };
}

// ── CLI entry point ──
const isCLI =
  typeof process !== "undefined" &&
  process.argv?.[1] &&
  import.meta.url.endsWith(process.argv[1].replace(/\\/g, "/"));
if (isCLI) {
  (async () => {
    const { readFileSync, writeFileSync } = await import("node:fs");
    const { parseArgs } = await import("node:util");
    const { join, dirname } = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const MODEL_PATH = join(__dirname, "output/runtime_model.json");
    const model = JSON.parse(readFileSync(MODEL_PATH, "utf8"));
    const command = process.argv[2];

    if (command === "add") {
      const { values } = parseArgs({
        args: process.argv.slice(3),
        options: {
          model: { type: "string" },
          vendorClass: { type: "string" },
          maxBufGB: { type: "string" },
          bwGBs: { type: "string" },
          tflops: { type: "string" },
          appleGen: { type: "string" },
        },
      });

      const name = values.model;
      const vc = values.vendorClass;
      const maxBufBytes = parseFloat(values.maxBufGB) * 1073741824;
      const bw = parseFloat(values.bwGBs);
      const tf = parseFloat(values.tflops);

      if (!name || !vc || !maxBufBytes || !bw || !tf) {
        console.error(
          "Missing required arguments. See usage in script header.",
        );
        process.exit(1);
      }
      if (vc === "apple" && !values.appleGen) {
        console.error("--appleGen is required when --vendorClass is 'apple'.");
        process.exit(1);
      }

      const { n, globalN, duplicate } = addGpu(model, {
        name,
        vendorClass: vc,
        maxBufBytes,
        bw,
        tf,
        appleGen: values.appleGen,
      });

      if (duplicate) {
        console.warn(
          `[duplicate-protection] GPU '${name}' is already recorded in class '${vc}'. Skipping addition.`,
        );
        process.exit(0);
      }

      console.log(
        `Added ${name} to ${vc} (n: ${n - 1} → ${n}, global: ${globalN})`,
      );

      updateCoefficients(model);
      model._meta.generated = new Date().toISOString();
      writeFileSync(MODEL_PATH, JSON.stringify(model, null, 2) + "\n");

      const vm = model.vendorModels[vc];
      if (vm) {
        console.log(
          `  bufToBw: a=${vm.bufToBw.a}, b=${vm.bufToBw.b}, r²=${vm.bufToBw.r2}`,
        );
        console.log(
          `  bwToTf:  a=${vm.bwToTf.a}, b=${vm.bwToTf.b}, r²=${vm.bwToTf.r2}`,
        );
        console.log(`  usable:  ${vm.regressionUsable}`);
      }
      console.log("✓ Updated runtime_model.json");
    } else if (command === "refit") {
      updateCoefficients(model);
      model._meta.generated = new Date().toISOString();
      writeFileSync(MODEL_PATH, JSON.stringify(model, null, 2) + "\n");
      console.log("✓ Re-derived all coefficients from sufficient stats.");
    } else {
      console.log("Usage:");
      console.log(
        "  node scripts/refit-model.mjs add --model NAME --vendorClass VC --maxBufGB N --bwGBs N --tflops N [--appleGen common-N]",
      );
      console.log("  node scripts/refit-model.mjs refit");
      process.exit(1);
    }
  })();
}
