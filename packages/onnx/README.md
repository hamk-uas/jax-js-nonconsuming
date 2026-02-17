# @hamk-uas/jax-js-nonconsuming-onnx

ONNX model loader for jax-js.

## Usage

Fetch a model from external path and load it.

```ts
import { ONNXModel } from "@hamk-uas/jax-js-nonconsuming-onnx";
import { numpy as np } from "@hamk-uas/jax-js-nonconsuming";

const modelBytes = await fetch("./model.onnx").then((r) => r.bytes());
const model = new ONNXModel(modelBytes);

try {
  using input = np.ones([1, 3, 224, 224]);
  const { output } = model.run({ input });
  output.dispose();
} finally {
  model.dispose();
}
```

Loaded models are ordinary functions and can be differentiated through. Use JIT when possible for
best performance.

```ts
import { grad, jit } from "@hamk-uas/jax-js-nonconsuming";

using run = jit(model.run);
const runGrad = grad((input: np.Array) => {
  const { output } = run({ input });
  using loss = computeLoss(output);
  return loss.mean();
});

using input = np.ones([1, 3, 224, 224]);
using dx = runGrad(input);
```

After you're done, you can free the model weights.

```ts
model.dispose();
```
