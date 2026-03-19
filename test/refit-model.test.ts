import { beforeEach, describe, expect, it } from "vitest";

// Import testable functions from the refit script (pure functions, no node:fs)
const { ols, welfordAdd, updateCoefficients, addGpu } = await import(
  // @ts-expect-error — plain .mjs has no type declarations
  "../scripts/refit-model.mjs"
);

// Minimal model fixture extracted from runtime_model.json — covers apple +
// discrete-modern vendor classes, all 4 Apple generations, Safari fallback,
// and sufficient stats.  Intentionally compact: tests exercise addGpu +
// updateCoefficients logic, not the full model schema.
const MODEL_FIXTURE = {
  _meta: {
    vendorClassCounts: {
      "discrete-modern": 24,
      "discrete-legacy": 2,
      apple: 14,
      igp: 7,
      mobile: 2,
    },
    trainingSet: "49 GPUs",
  },
  appleLookup: {
    "common-4": {
      n: 3,
      bw_range: [120, 400],
      tf_range: [3.62, 14.5],
      bw_geomean: 235.75882,
      tf_geomean: 7.246665,
      gpus: ["Apple_M4_Max_40C", "Apple_M4_Pro_20C", "Apple_M4_10C"],
    },
    "common-3": {
      n: 3,
      bw_range: [100, 400],
      tf_range: [3.05, 12.2],
      bw_geomean: 181.712058,
      tf_geomean: 5.893059,
      gpus: ["Apple_M3_Max_40C", "Apple_M3_Pro_18C", "Apple_M3_10C"],
    },
    "common-2": {
      n: 4,
      bw_range: [100, 800],
      tf_range: [3.58, 27.2],
      bw_geomean: 282.842725,
      tf_geomean: 9.74148,
      gpus: [
        "Apple_M2_Ultra_76C",
        "Apple_M2_Max_38C",
        "Apple_M2_Pro_19C",
        "Apple_M2_10C",
      ],
    },
    "common-1": {
      n: 4,
      bw_range: [68.2, 800],
      tf_range: [2.61, 20.8],
      bw_geomean: 257.03429,
      tf_geomean: 7.360971,
      gpus: [
        "Apple_M1_Ultra_64C",
        "Apple_M1_Max_32C",
        "Apple_M1_Pro_16C",
        "Apple_M1_8C",
      ],
    },
    _safari_fallback: {
      _doc: "P25 fallback",
      bw_p25: 120,
      tf_p25: 3.62,
      bw_median: 273,
      tf_median: 7.25,
    },
  },
  vendorModels: {
    "discrete-modern": {
      regressionUsable: true,
      bufToBw: { a: -12.026943, b: 0.784129, r2: 0.649214 },
      bwToTf: { a: -4.308606, b: 1.191885, r2: 0.549618 },
      bufToTf: { a: -24.972061, b: 1.206539, r2: 0.594685 },
      median: {},
      C_dispatch_us: 25,
      R_opt_words: 128,
      F_subgroup: 32,
    },
    apple: {
      regressionUsable: true,
      bufToBw: { a: -38.348665, b: 1.875667, r2: 0.615751 },
      bwToTf: { a: -3.099216, b: 0.934484, r2: 0.968269 },
      bufToTf: { a: -38.857177, b: 1.749432, r2: 0.593937 },
      median: {},
      C_dispatch_us: 20,
      R_opt_words: 128,
      F_subgroup: 32,
    },
  },
  globalFallback: {
    bufToBw: { a: -16.381322, b: 0.957754, r2: 0.651547 },
    bwToTf: { a: -4.512533, b: 1.207289, r2: 0.884679 },
    bufToTf: { a: -24.615406, b: 1.170468, r2: 0.590636 },
    median: {},
  },
  _sufficientStats: {
    perClass: {
      "discrete-modern": {
        n: 24,
        gpus: [],
        bufToBw: {
          mean_x: 23.271925,
          mean_y: 6.2212586,
          ssXX: 3.9512744,
          ssXY: 3.0983106,
          ssYY: 3.7421797,
        },
        bwToTf: {
          mean_x: 6.2212586,
          mean_y: 3.1064207,
          ssXX: 3.7421797,
          ssXY: 4.4602488,
          ssYY: 9.6723713,
        },
        bufToTf: {
          mean_x: 23.271925,
          mean_y: 3.1064207,
          ssXX: 3.9512744,
          ssXY: 4.767366,
          ssYY: 9.6723713,
        },
      },
      apple: {
        n: 14,
        gpus: [
          "Apple_M4_Max_40C",
          "Apple_M4_Pro_20C",
          "Apple_M4_10C",
          "Apple_M3_Max_40C",
          "Apple_M3_Pro_18C",
          "Apple_M3_10C",
          "Apple_M2_Ultra_76C",
          "Apple_M2_Max_38C",
          "Apple_M2_Pro_19C",
          "Apple_M2_10C",
          "Apple_M1_Ultra_64C",
          "Apple_M1_Max_32C",
          "Apple_M1_Pro_16C",
          "Apple_M1_8C",
        ],
        bufToBw: {
          mean_x: 23.368962,
          mean_y: 5.4837214,
          ssXX: 1.3727229,
          ssXY: 2.5747708,
          ssYY: 7.84312,
        },
        bwToTf: {
          mean_x: 5.4837214,
          mean_y: 2.025235,
          ssXX: 7.84312,
          ssXY: 7.3292722,
          ssYY: 7.0735397,
        },
        bufToTf: {
          mean_x: 23.368962,
          mean_y: 2.025235,
          ssXX: 1.3727229,
          ssXY: 2.4014855,
          ssYY: 7.0735397,
        },
      },
    },
    global: {
      n: 49,
      bufToBw: {
        mean_x: 22.978083,
        mean_y: 5.6260235,
        ssXX: 28.444679,
        ssXY: 27.242999,
        ssYY: 40.046333,
      },
      bwToTf: {
        mean_x: 5.6260235,
        mean_y: 2.2797042,
        ssXX: 40.046333,
        ssXY: 48.347507,
        ssYY: 65.978113,
      },
      bufToTf: {
        mean_x: 22.978083,
        mean_y: 2.2797042,
        ssXX: 28.444679,
        ssXY: 33.293586,
        ssYY: 65.978113,
      },
    },
    appleGenerations: {
      "common-4": { n: 3, sum_log_bw: 16.388428, sum_log_tf: 5.9416241 },
      "common-3": { n: 3, sum_log_bw: 15.60727, sum_log_tf: 5.3213256 },
      "common-2": { n: 4, sum_log_bw: 22.579564, sum_log_tf: 9.1055722 },
      "common-1": { n: 4, sum_log_bw: 22.196838, sum_log_tf: 7.9847676 },
    },
  },
};

function cloneModel() {
  return JSON.parse(JSON.stringify(MODEL_FIXTURE));
}

describe("refit-model", () => {
  describe("ols", () => {
    it("returns mean fallback for n < 2", () => {
      const w = { mean_x: 1, mean_y: 5, ssXX: 0, ssXY: 0, ssYY: 0 };
      const result = ols(1, w);
      expect(result).toEqual({ a: 5, b: 0, r2: 0 });
    });

    it("derives correct slope and intercept", () => {
      // Two points: (1, 2) and (3, 6) → y = x + slope=2, intercept=0 (in log)
      // Welford for [(1,2),(3,6)]: mean_x=2, mean_y=4, ssXX=2, ssXY=4, ssYY=8
      const w = { mean_x: 2, mean_y: 4, ssXX: 2, ssXY: 4, ssYY: 8 };
      const result = ols(2, w);
      expect(result.b).toBe(2); // slope
      expect(result.a).toBe(0); // intercept: 4 - 2*2 = 0
      expect(result.r2).toBe(1); // perfect fit
    });
  });

  describe("welfordAdd", () => {
    it("first point initializes correctly", () => {
      const w0 = { mean_x: 10, mean_y: 20, ssXX: 0, ssXY: 0, ssYY: 0 };
      // Adding a second point (14, 28) to a single-point start
      const w1 = welfordAdd(1, w0, 14, 28);
      expect(w1.mean_x).toBeCloseTo(12, 6);
      expect(w1.mean_y).toBeCloseTo(24, 6);
      expect(w1.ssXX).toBeCloseTo(8, 6); // (10-12)^2 + (14-12)^2 = 8, or dx*(x-mx_new)=4*2=8
      expect(w1.ssXY).toBeCloseTo(16, 6);
    });
  });

  describe("refit round-trip", () => {
    it("reproduces original coefficients", () => {
      const model = cloneModel();
      const before = JSON.parse(JSON.stringify(model.vendorModels));
      updateCoefficients(model);
      for (const vc of Object.keys(before)) {
        expect(model.vendorModels[vc].bufToBw).toEqual(before[vc].bufToBw);
        expect(model.vendorModels[vc].bwToTf).toEqual(before[vc].bwToTf);
        expect(model.vendorModels[vc].bufToTf).toEqual(before[vc].bufToTf);
      }
    });

    it("reproduces original Apple geomeans", () => {
      const model = cloneModel();
      const before = JSON.parse(JSON.stringify(model.appleLookup));
      updateCoefficients(model);
      for (const gen of ["common-1", "common-2", "common-3", "common-4"]) {
        expect(model.appleLookup[gen].bw_geomean).toEqual(
          before[gen].bw_geomean,
        );
        expect(model.appleLookup[gen].tf_geomean).toEqual(
          before[gen].tf_geomean,
        );
      }
    });
  });

  describe("addGpu — Apple new generation", () => {
    let model: ReturnType<typeof cloneModel>;

    beforeEach(() => {
      model = cloneModel();
    });

    it("creates structurally valid lookup entry for new generation", () => {
      addGpu(model, {
        name: "Apple_M5_12C",
        vendorClass: "apple",
        maxBufBytes: 16 * 1073741824,
        bw: 200,
        tf: 8.5,
        appleGen: "common-5",
      });
      updateCoefficients(model);

      const entry = model.appleLookup["common-5"];
      expect(entry).toBeDefined();
      // n must be populated
      expect(entry.n).toBe(1);
      // bw_range must be finite numbers, not [Infinity, 0]
      expect(entry.bw_range[0]).toBe(200);
      expect(entry.bw_range[1]).toBe(200);
      expect(Number.isFinite(entry.bw_range[0])).toBe(true);
      // tf_range must be finite numbers
      expect(entry.tf_range[0]).toBe(8.5);
      expect(entry.tf_range[1]).toBe(8.5);
      expect(Number.isFinite(entry.tf_range[0])).toBe(true);
      // gpus must contain the added model
      expect(entry.gpus).toEqual(["Apple_M5_12C"]);
      // geomeans must be derived (single point → exact value)
      expect(entry.bw_geomean).toBeCloseTo(200, 0);
      expect(entry.tf_geomean).toBeCloseTo(8.5, 0);
    });

    it("extends ranges when adding a second GPU to the same generation", () => {
      addGpu(model, {
        name: "Apple_M5_12C",
        vendorClass: "apple",
        maxBufBytes: 16 * 1073741824,
        bw: 200,
        tf: 8.5,
        appleGen: "common-5",
      });
      addGpu(model, {
        name: "Apple_M5_Max_40C",
        vendorClass: "apple",
        maxBufBytes: 32 * 1073741824,
        bw: 500,
        tf: 22,
        appleGen: "common-5",
      });
      updateCoefficients(model);

      const entry = model.appleLookup["common-5"];
      expect(entry.n).toBe(2);
      expect(entry.bw_range).toEqual([200, 500]);
      expect(entry.tf_range).toEqual([8.5, 22]);
      expect(entry.gpus).toEqual(["Apple_M5_12C", "Apple_M5_Max_40C"]);
      // Geomean of 200 and 500 ≈ 316.2
      expect(entry.bw_geomean).toBeCloseTo(Math.sqrt(200 * 500), 0);
    });

    it("does not duplicate GPU names on re-add", () => {
      addGpu(model, {
        name: "Apple_M5_12C",
        vendorClass: "apple",
        maxBufBytes: 16 * 1073741824,
        bw: 200,
        tf: 8.5,
        appleGen: "common-5",
      });
      const nAfterFirst = model._sufficientStats.perClass["apple"].n;
      const globalAfterFirst = model._sufficientStats.global.n;

      const result = addGpu(model, {
        name: "Apple_M5_12C",
        vendorClass: "apple",
        maxBufBytes: 16 * 1073741824,
        bw: 200,
        tf: 8.5,
        appleGen: "common-5",
      });
      expect(result.duplicate).toBe(true);
      expect(model.appleLookup["common-5"].gpus).toEqual(["Apple_M5_12C"]);
      // Stats must NOT be double-counted
      expect(model._sufficientStats.perClass["apple"].n).toBe(nAfterFirst);
      expect(model._sufficientStats.global.n).toBe(globalAfterFirst);
    });

    it("rejects re-add of seeded Apple GPU from checked-in model", () => {
      const nBefore = model._sufficientStats.perClass["apple"].n;
      const globalBefore = model._sufficientStats.global.n;

      // Apple_M4_10C is already in the seeded gpus list
      const result = addGpu(model, {
        name: "Apple_M4_10C",
        vendorClass: "apple",
        maxBufBytes: 16 * 1073741824,
        bw: 120,
        tf: 3.62,
        appleGen: "common-4",
      });
      expect(result.duplicate).toBe(true);
      expect(model._sufficientStats.perClass["apple"].n).toBe(nBefore);
      expect(model._sufficientStats.global.n).toBe(globalBefore);
    });

    it("preserves Safari fallback (not recomputed from generation geomeans)", () => {
      const beforeFb = JSON.parse(
        JSON.stringify(model.appleLookup._safari_fallback),
      );

      addGpu(model, {
        name: "Apple_M5_12C",
        vendorClass: "apple",
        maxBufBytes: 16 * 1073741824,
        bw: 200,
        tf: 8.5,
        appleGen: "common-5",
      });
      updateCoefficients(model);

      const fb = model.appleLookup._safari_fallback;
      // Safari fallback was calibrated from per-GPU P25/median (14 individual
      // Apple GPUs). updateCoefficients only has generation geomeans, so it
      // must NOT recompute — values should be unchanged.
      expect(fb.bw_p25).toBe(beforeFb.bw_p25);
      expect(fb.tf_p25).toBe(beforeFb.tf_p25);
      expect(fb.bw_median).toBe(beforeFb.bw_median);
      expect(fb.tf_median).toBe(beforeFb.tf_median);
    });

    it("updates existing generation metadata correctly", () => {
      const beforeN = model.appleLookup["common-4"].n;
      const beforeBwHi = model.appleLookup["common-4"].bw_range[1];

      addGpu(model, {
        name: "Apple_M4_Ultra_80C",
        vendorClass: "apple",
        maxBufBytes: 128 * 1073741824,
        bw: 800,
        tf: 28,
        appleGen: "common-4",
      });
      updateCoefficients(model);

      const entry = model.appleLookup["common-4"];
      expect(entry.n).toBe(beforeN + 1);
      expect(entry.bw_range[1]).toBe(Math.max(beforeBwHi, 800));
      expect(entry.gpus).toContain("Apple_M4_Ultra_80C");
      // Original GPUs preserved
      expect(entry.gpus).toContain("Apple_M4_Max_40C");
    });
  });

  describe("addGpu — non-Apple vendor", () => {
    it("increments class and global counts", () => {
      const model = cloneModel();
      const beforeClassN = model._sufficientStats.perClass["discrete-modern"].n;
      const beforeGlobalN = model._sufficientStats.global.n;

      addGpu(model, {
        name: "TEST_RTX_5090",
        vendorClass: "discrete-modern",
        maxBufBytes: 32 * 1073741824,
        bw: 1792,
        tf: 105,
      });

      expect(model._sufficientStats.perClass["discrete-modern"].n).toBe(
        beforeClassN + 1,
      );
      expect(model._sufficientStats.global.n).toBe(beforeGlobalN + 1);
    });

    it("does not create appleLookup entry for non-Apple GPU", () => {
      const model = cloneModel();
      const gensBefore = Object.keys(model.appleLookup).filter(
        (k) => !k.startsWith("_"),
      );

      addGpu(model, {
        name: "TEST_RTX_5090",
        vendorClass: "discrete-modern",
        maxBufBytes: 32 * 1073741824,
        bw: 1792,
        tf: 105,
      });
      updateCoefficients(model);

      const gensAfter = Object.keys(model.appleLookup).filter(
        (k) => !k.startsWith("_"),
      );
      expect(gensAfter).toEqual(gensBefore);
    });
  });

  describe("addGpu — Apple without appleGen", () => {
    it("throws when vendorClass is apple but appleGen is missing", () => {
      const model = cloneModel();
      expect(() =>
        addGpu(model, {
          name: "Apple_M5_12C",
          vendorClass: "apple",
          maxBufBytes: 16 * 1073741824,
          bw: 200,
          tf: 8.5,
        }),
      ).toThrow(/appleGen is required/);
    });
  });

  describe("addGpu — new vendor class", () => {
    it("creates vendor class from scratch", () => {
      const model = cloneModel();
      expect(model._sufficientStats.perClass["test-class"]).toBeUndefined();

      addGpu(model, {
        name: "TEST_GPU",
        vendorClass: "test-class",
        maxBufBytes: 4 * 1073741824,
        bw: 50,
        tf: 2,
      });

      const cs = model._sufficientStats.perClass["test-class"];
      expect(cs).toBeDefined();
      expect(cs.n).toBe(1);
      expect(Number.isFinite(cs.bufToBw.mean_x)).toBe(true);
    });
  });
});
