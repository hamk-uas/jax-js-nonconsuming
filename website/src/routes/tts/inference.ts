import { numpy as np, random, tree } from "@jax-js/jax";

import type { AudioPlayer } from "./audio";
import {
  createFlowLMState,
  createMimiDecodeState,
  type PocketTTS,
  runFlowLMStep,
  runMimiDecode,
} from "./pocket-tts";

export interface PlayTTSOptions {
  framesAfterEos: number;
  seed: number | null;
  lsdDecodeSteps: number;
  temperature: number;
  noiseClamp: number | null;
}

export async function playTTS(
  player: AudioPlayer,
  model: PocketTTS,
  embeds: np.Array,
  {
    framesAfterEos = 0,
    seed = null,
    lsdDecodeSteps = 1,
    temperature = 0.7,
    noiseClamp = null,
  }: Partial<PlayTTSOptions> = {},
): Promise<void> {
  let sequence = model.flowLM.bosEmb.reshape([1, -1]); // [1, 32]
  let audioPromise: Promise<void> = Promise.resolve();

  if (seed === null) seed = Math.floor(Math.random() * 2 ** 32);
  let key = random.key(seed);
  let flowLMState = createFlowLMState(model.flowLM);
  let mimiState = createMimiDecodeState(model.mimi);

  try {
    let eosStep: number | null = null;

    console.log("Starting TTS generation...");
    let lastTimestamp = performance.now();

    for (let step = 0; step < 1000; step++) {
      const oldKey = key;
      let stepKey: np.Array;
      [key, stepKey] = random.split(oldKey);
      oldKey.dispose();
      const {
        latent,
        isEos,
        state: newFlowLMState,
      } = runFlowLMStep(
        model.flowLM,
        flowLMState,
        stepKey,
        step === 0 ? sequence : sequence.slice([-1]),
        step === 0 ? embeds : null,
        flowLMState.kvCacheLen, // same as offset
        lsdDecodeSteps,
        temperature,
        noiseClamp,
      );
      flowLMState = newFlowLMState;
      stepKey.dispose();

      const isEosData = await isEos.data();
      isEos.dispose();
      if (isEosData[0] && eosStep === null) {
        console.log(`🛑 EOS at step ${step}!`);
        eosStep = step;
      }
      if (eosStep !== null && step >= eosStep + framesAfterEos) {
        console.log(
          `Generation ended at step ${step}, ${framesAfterEos} frames after EOS.`,
        );
        latent.dispose();
        break;
      }

      {
        const oldSeq = sequence;
        sequence = np.concatenate([oldSeq, latent]);
        oldSeq.dispose();
      }
      latent.dispose();

      const timestamp = performance.now();
      console.log(
        `Generated step ${step} in ${(timestamp - lastTimestamp).toFixed(1)} ms`,
      );
      lastTimestamp = timestamp;

      using mimiInputSlice = sequence.slice([-1]);
      using mimiInputScaled = mimiInputSlice.mul(model.flowLM.embStd);
      using mimiInput = mimiInputScaled.add(model.flowLM.embMean);

      const [audio, newMimiState] = runMimiDecode(
        model.mimi,
        mimiState,
        mimiInput,
        step,
      );
      mimiState = newMimiState;

      const lastAudioPromise = audioPromise;
      audioPromise = (async () => {
        using sliced = audio.slice(0);
        using clipped = np.clip(sliced, -1, 1);
        using asFloat = clipped.astype(np.float32);
        const audioPcm = (await asFloat.data()) as Float32Array;
        audio.dispose();
        if (audioPcm.length !== 1920) {
          throw new Error(
            `expected 1920 audio samples, got ${audioPcm.length}`,
          );
        }
        await lastAudioPromise;
        await player.playChunk(audioPcm);
      })();
    }
  } finally {
    key.dispose();
    sequence.dispose();
    tree.dispose(flowLMState);
    tree.dispose(mimiState);
    tree.dispose([model, embeds]);
    await audioPromise;
  }
}
