// Tests for convolution-related operations.

import {
  defaultDevice,
  devices,
  grad,
  init,
  jit,
  lax,
  numpy as np,
  vmap,
} from "@hamk-uas/jax-js-nonconsuming";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("1d convolution", () => {
    using x = np.array([[[1, 2, 3, 4, 5]]]);
    using y = np.array([[[2, 0.5, -1]]]);
    using result = lax.convGeneralDilated(x, y, [1], "VALID");
    expect(result.js()).toEqual([[[0, 1.5, 3]]]);

    using result2 = lax.convGeneralDilated(x, y, [1], "SAME");
    expect(result2.js()).toEqual([[[-1.5, 0, 1.5, 3, 10.5]]]);
  });

  test("padding 'SAME' and 'SAME_LOWER'", () => {
    using x = np.ones([1, 1, 5]);
    using y = np.ones([1, 1, 4]);
    using resultSame = lax.convGeneralDilated(x, y, [1], "SAME");
    {
      using _s = resultSame.slice(0, 0);
      expect(_s.js()).toEqual([3, 4, 4, 3, 2]);
    }
    using resultSameLower = lax.convGeneralDilated(x, y, [1], "SAME_LOWER");
    {
      using _s = resultSameLower.slice(0, 0);
      expect(_s.js()).toEqual([2, 3, 4, 4, 3]);
    }
  });

  test("2d convolution", () => {
    const _rawX = np.array([
      [3, 1, 5],
      [2, 2, 9],
    ]);
    using x = _rawX.reshape([1, 1, 2, 3]);
    _rawX.dispose();
    const _rawY = np.array([
      [1, 2],
      [3, 4],
    ]);
    using y = _rawY.reshape([1, 1, 2, 2]);
    _rawY.dispose();
    using result = lax.convGeneralDilated(x, y, [1, 1], "VALID");
    {
      using _s = result.slice(0, 0);
      expect(_s.js()).toEqual([[19, 53]]);
    }
  });

  test("conv works with jit", () => {
    using convFn = jit((a: np.Array, b: np.Array) =>
      lax.convGeneralDilated(a, b, [1], "SAME"),
    );
    using x = np.array([[[1, 2, 3, 4, 5]]]);
    using y = np.array([[[2, 0.5, -1]]]);
    using result = convFn(x, y);
    expect(result.js()).toEqual([[[-1.5, 0, 1.5, 3, 10.5]]]);
  });

  test("0d convolution", () => {
    using x = np.array([
      [1, 2],
      [3, 4],
      [5, 8],
    ]);
    using y = np.array([
      [6, 4],
      [3, 2],
    ]);
    using result = lax.convGeneralDilated(x, y, [], "VALID");
    expect(result.js()).toEqual([
      [14, 7],
      [34, 17],
      [62, 31],
    ]);
  });

  test("grad of 0d convolution", () => {
    using x = np.array([
      [1, 2],
      [3, 4],
      [5, 8],
    ]);
    using y = np.array([
      [6, 4],
      [3, 2],
    ]);
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [], "VALID").sum();
    {
      using _gr = grad(f)(x, y);
      expect(_gr.js()).toEqual([
        [9, 6],
        [9, 6],
        [9, 6],
      ]);
    }
  });

  test("grad of 1d convolution", () => {
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [1], "SAME").slice(0, 0, 3);
    using x = np.array([[[1, 2, 3, 4, 5, 6, 7]]]);
    using y = np.array([[[2, 0.5, -1]]]);
    using dx = grad(f)(x, y);
    {
      using _s = dx.slice(0, 0);
      expect(_s.js()).toEqual([0, 0, 2, 0.5, -1, 0, 0]);
    }

    using dy = grad((y: np.Array, x: np.Array) => f(x, y))(y, x);
    {
      using _s = dy.slice(0, 0);
      expect(_s.js()).toEqual([3, 4, 5]);
    }
  });

  test("grad shape test with stride 2", () => {
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [2, 2], "VALID").sum();
    const g = (y: np.Array, x: np.Array) =>
      lax.convGeneralDilated(x, y, [2, 2], "VALID").sum();

    for (const xDim of [1, 3, 8, 12, 15]) {
      for (const kDim of [1, 3, 4]) {
        if (xDim < kDim) continue;
        using x = np.zeros([3, 1, xDim, xDim]);
        using y = np.zeros([1, 1, kDim, kDim]);
        using dx = grad(f)(x, y);
        expect(dx.shape).toEqual(x.shape);

        using dy = grad(g)(y, x);
        expect(dy.shape).toEqual(y.shape);
      }
    }
  });

  test("max-pooling and min-pooling", () => {
    using x = np.array([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    using result = lax.reduceWindow(x, np.max, [2, 2], [1, 2]);
    expect(result.js()).toEqual([
      [6, 8],
      [10, 12],
    ]);

    using resultMin = lax.reduceWindow(x, np.min, [2, 2], [1, 2]);
    expect(resultMin.js()).toEqual([
      [1, 3],
      [5, 7],
    ]);
  });

  test("grad of max-pool 2d", () => {
    using x = np.array([
      [1, 5, 3, 4],
      [1, 2, 3, 4],
    ]);
    const maxPool2x2Sum = (x: np.Array) =>
      lax.reduceWindow(x, np.max, [2, 2], [2, 2]).sum();

    {
      using _jit = jit(maxPool2x2Sum);
      using _r = _jit(x);
      expect(_r.js()).toEqual(9);
    } // 5 + 4
    {
      using _gr = grad(maxPool2x2Sum)(x);
      expect(_gr.js()).toEqual([
        [0, 1, 0, 0.5],
        [0, 0, 0, 0.5],
      ]);
    }
  });

  test("grouped convolution shape", () => {
    // Test with 2 groups: input has 4 channels, output has 6 channels
    // Each group: 2 input channels -> 3 output channels
    using x = np.zeros([2, 4, 8, 8]); // [N, C_in, H, W]
    using y = np.zeros([6, 2, 3, 3]); // [C_out, C_in/G, kH, kW]
    using result = lax.convGeneralDilated(x, y, [1, 1], "VALID", {
      featureGroupCount: 2,
    });
    expect(result.shape).toEqual([2, 6, 6, 6]);

    // Test with 4 groups (depthwise-like): 4 channels, each convolved separately
    using x2 = np.zeros([1, 4, 5, 5]);
    using y2 = np.zeros([8, 1, 3, 3]); // 2 output channels per group
    using result2 = lax.convGeneralDilated(x2, y2, [1, 1], "SAME", {
      featureGroupCount: 4,
    });
    expect(result2.shape).toEqual([1, 8, 5, 5]);
  });

  test("grouped convolution values", () => {
    // 2 groups, each doing independent 1d convolution
    // Group 1: channel 0 with kernel 0
    // Group 2: channel 1 with kernel 1
    const _rawX = np.array([
      [[1, 2, 3, 4]], // channel 0
      [[5, 6, 7, 8]], // channel 1
    ]);
    using x = _rawX.reshape([1, 2, 4]); // [N=1, C_in=2, W=4]
    _rawX.dispose();

    const _rawY = np.array([
      [[1, 0, -1]], // kernel for group 0 -> out channel 0
      [[1, 1, 1]], // kernel for group 1 -> out channel 1
    ]);
    using y = _rawY.reshape([2, 1, 3]); // [C_out=2, C_in/G=1, kW=3]
    _rawY.dispose();

    using result = lax.convGeneralDilated(x, y, [1], "VALID", {
      featureGroupCount: 2,
    });
    // Group 0: [1,2,3,4] conv [1,0,-1] = [1-3, 2-4] = [-2, -2]
    // Group 1: [5,6,7,8] conv [1,1,1] = [5+6+7, 6+7+8] = [18, 21]
    expect(result.shape).toEqual([1, 2, 2]);
    expect(result.js()).toEqual([
      [
        [-2, -2],
        [18, 21],
      ],
    ]);
  });

  test("grad of depthwise conv1d", () => {
    // Depthwise conv: each input channel convolved with its own kernel
    // 3 input channels, 3 output channels (1 per group)
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [1], "VALID", { featureGroupCount: 3 });

    const _rawX = np.array([
      [[1, 2, 3, 4, 5]], // channel 0
      [[2, 3, 4, 5, 6]], // channel 1
      [[3, 4, 5, 6, 7]], // channel 2
    ]);
    using x = _rawX.reshape([1, 3, 5]); // [N=1, C=3, W=5]
    _rawX.dispose();

    const _rawY = np.array([
      [[1, -1]], // kernel for channel 0
      [[1, 0]], // kernel for channel 1
      [[0, 1]], // kernel for channel 2
    ]);
    using y = _rawY.reshape([3, 1, 2]); // [C_out=3, C_in/G=1, kW=2]
    _rawY.dispose();

    // Forward pass check
    using result = f(x, y);
    expect(result.shape).toEqual([1, 3, 4]);
    // Channel 0: [1-2, 2-3, 3-4, 4-5] = [-1, -1, -1, -1]
    // Channel 1: [2, 3, 4, 5]
    // Channel 2: [4, 5, 6, 7]
    expect(result.js()).toEqual([
      [
        [-1, -1, -1, -1],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
      ],
    ]);

    // Gradient w.r.t. input
    const sumF = (x: np.Array, y: np.Array) => f(x, y).sum();
    using dx = grad(sumF)(x, y);
    expect(dx.shape).toEqual([1, 3, 5]);

    // Gradient w.r.t. kernel
    using dy = grad((y: np.Array, x: np.Array) => sumF(x, y))(y, x);
    expect(dy.shape).toEqual([3, 1, 2]);
    // dy[0] = sum of x[0] windows = [1+2+3+4, 2+3+4+5] = [10, 14]
    // dy[1] = sum of x[1] windows = [2+3+4+5, 3+4+5+6] = [14, 18]
    // dy[2] = sum of x[2] windows = [3+4+5+6, 4+5+6+7] = [18, 22]
    expect(dy.js()).toEqual([[[10, 14]], [[14, 18]], [[18, 22]]]);
  });

  test("vmapped 1d convolution", () => {
    // vmap over a batch of inputs with a single kernel
    // lhs shape: [N, C_in, W], rhs shape: [C_out, C_in, kW]
    const conv1d = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [1], "VALID");

    // 3 different inputs to vmap over, each with shape [1, 1, 5]
    using x = np.array([
      [[[1, 2, 3, 4, 5]]], // input 0: [N=1, C=1, W=5]
      [[[2, 3, 4, 5, 6]]], // input 1
      [[[3, 4, 5, 6, 7]]], // input 2
    ]); // shape [3, 1, 1, 5]

    using y = np.array([[[2, 0.5, -1]]]); // shape [1, 1, 3] = [C_out=1, C_in=1, kW=3]

    // vmap over x (axis 0), keep y unbatched (null)
    const vmappedConv = vmap(conv1d, [0, null]);
    using result = vmappedConv(x, y);

    // Each input is convolved with the same kernel
    // [1,2,3,4,5] conv [2,0.5,-1] = [2+1-3, 4+1.5-4, 6+2-5] = [0, 1.5, 3]
    // [2,3,4,5,6] conv [2,0.5,-1] = [4+1.5-4, 6+2-5, 8+2.5-6] = [1.5, 3, 4.5]
    // [3,4,5,6,7] conv [2,0.5,-1] = [6+2-5, 8+2.5-6, 10+3-7] = [3, 4.5, 6]
    expect(result.shape).toEqual([3, 1, 1, 3]);
    expect(result.js()).toEqual([
      [[[0, 1.5, 3]]],
      [[[1.5, 3, 4.5]]],
      [[[3, 4.5, 6]]],
    ]);
  });

  test("vmapped 2d convolution over inputs and kernels", () => {
    // vmap over both inputs and kernels
    const conv2d = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [1, 1], "VALID");

    // 2 different inputs, each with shape [N=1, C_in=1, H=2, W=3]
    using x = np.array([
      [
        [
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
        ],
      ], // input 0
      [
        [
          [
            [2, 3, 4],
            [5, 6, 7],
          ],
        ],
      ], // input 1
    ]); // shape [2, 1, 1, 2, 3]

    // 2 different kernels, each with shape [C_out=1, C_in=1, kH=2, kW=2]
    using y = np.array([
      [
        [
          [
            [1, 0],
            [0, 1],
          ],
        ],
      ], // kernel 0
      [
        [
          [
            [0, 1],
            [1, 0],
          ],
        ],
      ], // kernel 1
    ]); // shape [2, 1, 1, 2, 2]

    // vmap over both x and y (axis 0)
    const vmappedConv = vmap(conv2d, [0, 0]);
    using result = vmappedConv(x, y);

    // input 0 conv kernel 0: [[1+5, 2+6]] = [[6, 8]]
    // input 1 conv kernel 1: [[3+5, 4+6]] = [[8, 10]]
    expect(result.shape).toEqual([2, 1, 1, 1, 2]);
    expect(result.js()).toEqual([[[[[6, 8]]]], [[[[8, 10]]]]]);
  });

  function checkConvTransposeShape(
    xShape: number[],
    kShape: number[],
    strides: number[],
    padding: lax.PaddingType,
    expectedShape: number[],
  ) {
    using x = np.zeros(xShape);
    using k = np.zeros(kShape);
    using result = lax.convTranspose(x, k, strides, padding);
    expect(result.shape).toEqual(expectedShape);
    result.dispose();
  }

  test("convTranspose shape tests", () => {
    // 1D tests
    // SAME padding: output spatial = input spatial * stride
    checkConvTransposeShape([1, 1, 4], [1, 1, 3], [2], "SAME", [1, 1, 8]);
    checkConvTransposeShape([1, 1, 5], [1, 1, 3], [2], "SAME", [1, 1, 10]);
    checkConvTransposeShape([2, 3, 6], [5, 3, 4], [3], "SAME", [2, 5, 18]);

    // VALID padding: output = (input - 1) * stride + kernel
    checkConvTransposeShape([1, 1, 4], [1, 1, 3], [2], "VALID", [1, 1, 9]);
    checkConvTransposeShape([1, 1, 5], [1, 1, 3], [2], "VALID", [1, 1, 11]);
    checkConvTransposeShape([2, 3, 6], [5, 3, 4], [3], "VALID", [2, 5, 19]);

    // 2D tests
    // SAME padding
    checkConvTransposeShape(
      [1, 1, 4, 4],
      [1, 1, 3, 3],
      [2, 2],
      "SAME",
      [1, 1, 8, 8],
    );
    checkConvTransposeShape(
      [2, 3, 8, 8],
      [5, 3, 3, 3],
      [2, 2],
      "SAME",
      [2, 5, 16, 16],
    );
    checkConvTransposeShape(
      [1, 2, 5, 7],
      [4, 2, 4, 4],
      [2, 3],
      "SAME",
      [1, 4, 10, 21],
    );

    // VALID padding
    checkConvTransposeShape(
      [1, 1, 4, 4],
      [1, 1, 3, 3],
      [2, 2],
      "VALID",
      [1, 1, 9, 9],
    );
    checkConvTransposeShape(
      [1, 2, 5, 5],
      [4, 2, 4, 4],
      [2, 2],
      "VALID",
      [1, 4, 12, 12],
    );
  });

  test("convTranspose 1d 2x upscale", () => {
    // 2x upscaling with stride 2 and kernel [1, 1]
    // Stretched input: [1, 0, 2, 0, 3, 0] conv [1, 1] -> [1, 1, 2, 2, 3, 3]
    using x = np.array([[[1, 2, 3]]]); // [N=1, C=1, W=3]
    using k = np.array([[[1, 1]]]); // [C_out=1, C_in=1, kW=2]

    using result = lax.convTranspose(x, k, [2], "SAME", {});
    expect(result.shape).toEqual([1, 1, 6]);
    {
      using _s = result.slice(0, 0);
      expect(_s.js()).toEqual([1, 1, 2, 2, 3, 3]);
    }
  });
});
