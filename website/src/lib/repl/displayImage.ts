import { numpy as np } from "@jax-js-nonconsuming/jax";

/**
 * Convert a jax-js array (2D grayscale, 3D RGB/RGBA) to a data URL.
 *
 * Supports:
 * - 2D (H, W) → grayscale
 * - 3D (H, W, 1) → grayscale
 * - 3D (H, W, 3) → RGB
 * - 3D (H, W, 4) → RGBA
 *
 * Float inputs are treated as [0, 1) and scaled to [0, 255].
 * Bool inputs are converted to 0 or 255.
 */
export async function arrayToDataUrl(ar: np.Array): Promise<string> {
  if (ar.ndim !== 2 && ar.ndim !== 3) {
    throw new Error(
      "displayImage() only supports 2D (H, W) or 3D (H, W, C) array",
    );
  }
  await ar.blockUntilReady();

  if (ar.ndim === 2) {
    // If 2D, convert to (H, W, 1)
    using old = ar;
    ar = old.reshape([...old.shape, 1]);
  }
  const height = ar.shape[0];
  const width = ar.shape[1];
  const channels = ar.shape[2];

  if (ar.dtype === np.float32 || ar.dtype === np.float16) {
    // If float, normalize [0, 1) to [0, 256)
    using old = ar;
    ar = np.clip(old.mul(256), 0, 255).astype(np.uint32);
  } else if (ar.dtype === np.bool) {
    // If bool, convert to 0 or 255
    using old = ar;
    ar = old.astype(np.uint32).mul(255);
  }

  let rgbaArray: np.Array;
  if (channels === 1) {
    using expanded = np.repeat(ar, 3, 2);
    using alphas = np.full([height, width, 1], 255, {
      dtype: ar.dtype,
      device: ar.device,
    });
    rgbaArray = np.concatenate([expanded, alphas], 2);
  } else if (channels === 3) {
    using alphas = np.full([height, width, 1], 255, {
      dtype: ar.dtype,
      device: ar.device,
    });
    rgbaArray = np.concatenate([ar, alphas], 2);
  } else if (channels === 4) {
    rgbaArray = ar;
  } else {
    throw new Error(
      "displayImage() only supports 1, 3, or 4 channels in the last dimension",
    );
  }
  ar.dispose();

  const buf = await rgbaArray.data();
  rgbaArray.dispose();
  const dataArray = new Uint8ClampedArray(buf);
  const imageData = new ImageData(dataArray, width, height, {
    colorSpace: "srgb",
  });

  // Create a temporary <canvas> to draw and produce a data URL.
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}
