import { beforeEach, expect, suite, test } from "vitest";

import { defaultDevice, devices, init } from "../backend";
import { arange, array, eye, ones, zeros } from "./array";
import { DType } from "../alu";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);

  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("can construct zeros()", async () => {
    using ar = zeros([3, 3]);
    expect(ar.shape).toEqual([3, 3]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    using t = ar.transpose();
    expect(await t.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(t.dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]));
  });

  test("can construct ones()", async () => {
    using ar = ones([2, 2]);
    expect(ar.shape).toEqual([2, 2]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  test("can add two arrays", async () => {
    using ar1 = ones([2, 2]);
    using ar2 = ones([2, 2]);
    using ar3 = ar1.add(ar2);
    expect(ar3.shape).toEqual([2, 2]);
    expect(ar3.dtype).toEqual("float32");
    expect(await ar3.data()).toEqual(new Float32Array([2, 2, 2, 2]));
  });

  test("can construct arrays from data", () => {
    using a = array([1, 2, 3, 4]);
    using b = array([10, 5, 2, -8.5]);
    using c = a.mul(b);
    expect(c.shape).toEqual([4]);
    expect(c.dtype).toEqual("float32");
    expect(c.dataSync()).toEqual(new Float32Array([10, 10, 6, -34]));
    using r = c.reshape([2, 2]);
    using rt = r.transpose();
    expect(rt.dataSync()).toEqual(new Float32Array([10, 6, 10, -34]));
  });

  test("common broadcasting", () => {
    // Start with arrays of shape [2, 2] and [2, 3].
    // Reshape first one to [2, 1, 2] and second one to [2, 3, 1].
    using rawA = array([
      [1, 22],
      [3, 9],
    ]);
    using a = rawA.reshape([2, 1, 2]);
    using rawB = array([
      [10, 5, -2],
      [-8, 0, 3],
    ]);
    using b = rawB.reshape([2, 3, 1]);

    // Multiply them together -- outer products of a[i] and b[i].
    using c = a.mul(b);
    expect(c.shape).toEqual([2, 3, 2]);
    expect(c.js()).toEqual([
      [
        [10, 220],
        [5, 110],
        [-2, -44],
      ],
      [
        [-24, -72],
        [0, 0],
        [9, 27],
      ],
    ]);
  });

  test("flatten and ravel", () => {
    using a = array([
      [
        [1, 2],
        [3, 4],
      ],
    ]); // 3D
    expect(a.shape).toEqual([1, 2, 2]);
    using flat = a.flatten();
    expect(flat.js()).toEqual([1, 2, 3, 4]);
    using rav = a.ravel();
    expect(rav.js()).toEqual([1, 2, 3, 4]);
    using s = array(3);
    using sf = s.flatten();
    expect(sf.js()).toEqual([3]);
  });

  test("can add array to itself", () => {
    using a = array([1, 2, 3]);
    // Make sure duplicate references don't trip up the backend.
    using ab = a.add(a);
    using b = ab.add(a);
    expect(b.dataSync()).toEqual(new Float32Array([3, 6, 9]));
  });

  test("can coerce array to primitive", () => {
    using a = array(42);
    expect(a).toBeCloseTo(42);

    // https://github.com/microsoft/TypeScript/issues/42218
    expect(+(a as any)).toEqual(42);
    expect((a as any) + 1).toEqual(43);
    expect((a as any) ** 2).toEqual(42 ** 2);
  });

  test("construct bool array", () => {
    using a = array([true, false, true]);
    expect(a.shape).toEqual([3]);
    expect(a.dtype).toEqual("bool");

    expect(a.dataSync()).toEqual(new Int32Array([1, 0, 1]));
    expect(a.js()).toEqual([true, false, true]);

    using b = array([1, 3, 4]);
    using g = b.greater(2);
    expect(g.js()).toEqual([false, true, true]);
    expect(g.dataSync()).toEqual(new Int32Array([0, 1, 1]));

    using eq = b.equal(3);
    expect(eq.js()).toEqual([false, true, false]);
    using cmpArr = array([2, 3, 4]);
    using ne = b.notEqual(cmpArr);
    expect(ne.js()).toEqual([true, false, false]);
  });

  test("comparison operators async", { timeout: 30_000 }, async () => {
    using x = array([1, 2, 3]);
    using g = x.greater(2);
    expect(await g.jsAsync()).toEqual([false, false, true]);
    using ge = x.greaterEqual(2);
    expect(await ge.jsAsync()).toEqual([false, true, true]);
    using l = x.less(2);
    expect(await l.jsAsync()).toEqual([true, false, false]);
    using le = x.lessEqual(2);
    expect(await le.jsAsync()).toEqual([true, true, false]);
    using eq = x.equal(2);
    expect(await eq.jsAsync()).toEqual([false, true, false]);
    using ne = x.notEqual(2);
    expect(await ne.jsAsync()).toEqual([true, false, true]);

    using ar1 = arange(0, 5000, 1, { dtype: DType.Float32 });
    await ar1.data(); // Ensure data is loaded
    using ar = ar1.add(1);
    using cmp = ar.less(2500);
    const vals = (await cmp.data()) as Int32Array;
    for (let i = 0; i < vals.length; i++) {
      expect(vals[i]).toEqual(i + 1 < 2500 ? 1 : 0);
    }
  });

  test("comparison ops handle nan", async () => {
    using x = array([NaN, 0]);
    using g = x.greater(NaN);
    expect(await g.jsAsync()).toEqual([false, false]);
    using l = x.less(NaN);
    expect(await l.jsAsync()).toEqual([false, false]);
    using eq = x.equal(NaN);
    expect(await eq.jsAsync()).toEqual([false, false]);
    using ne = x.notEqual(NaN);
    expect(await ne.jsAsync()).toEqual([true, true]);
    using ge = x.greaterEqual(NaN);
    expect(await ge.jsAsync()).toEqual([false, false]);
    using le = x.lessEqual(NaN);
    expect(await le.jsAsync()).toEqual([false, false]);
  });

  test("slicing arrays", () => {
    using x = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Basic slicing and element access.
    using s1 = x.slice(0, 0);
    expect(s1.js()).toEqual(1);
    using s2 = x.slice(0, 2);
    expect(s2.js()).toEqual(3);
    using s3 = x.slice(1, 2);
    expect(s3.js()).toEqual(6);
    using s4 = x.slice(1);
    expect(s4.js()).toEqual([4, 5, 6]);
    using s5 = x.slice();
    expect(s5.js()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Try slicing with negative indices.
    using s6 = x.slice(-1, -1);
    expect(s6.js()).toEqual(6);
    using s7 = x.slice(-2, -1);
    expect(s7.js()).toEqual(3);
    using s8 = x.slice(-1, -3);
    expect(s8.js()).toEqual(4);

    // Try adding new axes.
    using s9 = x.slice(0, 0, null);
    expect(s9.js()).toEqual([1]);
    using s10 = x.slice(0, null, 0);
    expect(s10.js()).toEqual([1]);
    using s11 = x.slice(null);
    expect(s11.js()).toEqual([x.js()]);
  });

  test("sum along negative axis", () => {
    using x = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using s1 = x.sum(-1);
    expect(s1.js()).toEqual([6, 15]);
    using s2 = x.sum(-2);
    expect(s2.js()).toEqual([5, 7, 9]);
  });

  test("mean along multiple axes", () => {
    using x = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    using m1 = x.mean();
    expect(m1.js()).toEqual(3.5);
    using m2 = x.mean([0, 1]);
    expect(m2.js()).toEqual(3.5);
    using m3 = x.mean(0);
    expect(m3.js()).toEqual([2.5, 3.5, 4.5]);
    using m4 = x.mean(1);
    expect(m4.js()).toEqual([2, 5]);
  });

  test("advanced indexing with gather", () => {
    using x = array([1, 3, 2], { dtype: DType.Int32 });
    // np.eye(5)[[1, 3, 2]]
    using e1 = eye(5, { dtype: DType.Float32 });
    using g1 = e1.slice(x);
    expect(g1.js()).toEqual([
      [0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 1, 0, 0],
    ]);
    // np.eye(5)[[1, 3, 2], np.newaxis]
    using e2 = eye(5, { dtype: DType.Float32 });
    using g2 = e2.slice(x, null);
    expect(g2.js()).toEqual([
      [[0, 1, 0, 0, 0]],
      [[0, 0, 0, 1, 0]],
      [[0, 0, 1, 0, 0]],
    ]);
    // np.eye(5)[1:4, [1, 3, 2]]
    using e3 = eye(5, { dtype: DType.Float32 });
    using g3 = e3.slice([1, 4], x);
    expect(g3.js()).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]);
  });

  // This checks to make sure that index calculations don't suddenly break for
  // large arrays, covering workgroups > 65535 in WebGPU for instance.
  if (device !== "cpu" && device !== "webgl") {
    test("large array dispatch", async () => {
      using x = ones([100, 1000, 1000], { dtype: DType.Int32 }); // 100M elements
      await x.blockUntilReady();
      using s = x.sum();
      expect(await s.jsAsync()).toEqual(100_000_000);
    });
  }

  test("iterate over an array", () => {
    using src = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const [a, b] = src;
    expect(a.js()).toEqual([1, 2, 3]);
    expect(b.js()).toEqual([4, 5, 6]);
    a.dispose();
    b.dispose();

    using raw = array([1, 2, 3, 4]);
    using reshaped = raw.reshape([2, 2]);
    const [row0, z] = reshaped;
    const [x, y] = row0;
    expect(x.js()).toEqual(1);
    expect(y.js()).toEqual(2);
    expect(z.js()).toEqual([3, 4]);
    x.dispose();
    y.dispose();
    row0.dispose();
    z.dispose();
  });

  test("u32 data type", () => {
    using a = array([1, 2, 3], { dtype: DType.Uint32 });
    expect(a.dtype).toBe(DType.Uint32);
    expect(a.dataSync()).toEqual(new Uint32Array([1, 2, 3]));
    expect(a.js()).toEqual([1, 2, 3]);

    using sub = array(2, { dtype: DType.Uint32 });
    using b = a.sub(sub);
    expect(b.dtype).toBe(DType.Uint32);
    expect(b.dataSync()).toEqual(new Uint32Array([4294967295, 0, 1]));
    expect(b.js()).toEqual([4294967295, 0, 1]);
  });

  test("casting arrays", () => {
    using a = array([1, 2, 3], { dtype: DType.Int32 });
    expect(a.dtype).toBe(DType.Int32);
    expect(a.dataSync()).toEqual(new Int32Array([1, 2, 3]));
    expect(a.js()).toEqual([1, 2, 3]);

    using b = a.astype(DType.Float32);
    expect(b.dtype).toBe(DType.Float32);
    expect(b.dataSync()).toEqual(new Float32Array([1, 2, 3]));
    expect(b.js()).toEqual([1, 2, 3]);
  });

  test("cast saturates from large f32 -> i32", () => {
    using a = array([1e20, -1e20, 1e10, -1e10, 1e5, -1e5], {
      dtype: DType.Float32,
    });
    using b = a.astype(DType.Int32);
    expect(b.js()).toEqual([
      2147483647, -2147483648, 2147483647, -2147483648, 100000, -100000,
    ]);
  });

  test("cast saturates from large f32 -> u32", () => {
    using a = array([1e20, -1e20, 1e10, -1e10, 1e5, -1e5], {
      dtype: DType.Float32,
    });
    using b = a.astype(DType.Uint32);
    expect(b.js()).toEqual([4294967295, 0, 4294967295, 0, 100000, 0]);
  });

  test("view float32 -> int32 bitcast", () => {
    // IEEE 754: 1.0f = 0x3F800000 = 1065353216
    using a = array([1.0, 2.0], { dtype: DType.Float32 });
    using b = a.view(DType.Int32);
    expect(b.dtype).toBe(DType.Int32);
    expect(b.js()).toEqual([1065353216, 1073741824]);

    // Round-trip back to float32.
    using c = b.view(DType.Float32);
    expect(c.dtype).toBe(DType.Float32);
    expect(c.js()).toEqual([1.0, 2.0]);
  });

  test("view int32 -> float32 bitcast", () => {
    using a = array([1065353216, 1073741824], { dtype: DType.Int32 });
    using b = a.view(DType.Float32);
    expect(b.dtype).toBe(DType.Float32);
    expect(b.js()).toEqual([1.0, 2.0]);
  });

  test("view identity and invalid values", () => {
    using a = array([1, 2, 3], { dtype: DType.Float32 });
    using b = a.view(DType.Float32);
    expect(b.dtype).toBe(DType.Float32);
    expect(b.js()).toEqual([1, 2, 3]);

    // Bool cannot be cast currently.
    using c = array([true, false]);
    expect(() => c.view(DType.Int32)).toThrow();

    using d = array([1, 0], { dtype: DType.Int32 });
    expect(() => d.view(DType.Bool)).toThrow();
  });

  test("view preserves special float values", () => {
    using a = array([Infinity, -Infinity, NaN], { dtype: DType.Float32 });
    using b = a.view(DType.Int32);
    expect(b.dtype).toBe(DType.Int32);
    // Round-trip through int32 and back.
    using c = b.view(DType.Float32);
    expect(c.dtype).toBe(DType.Float32);
    const vals = c.js() as number[];
    expect(vals[0]).toBe(Infinity);
    expect(vals[1]).toBe(-Infinity);
    expect(vals[2]).toBeNaN();
  });
});
