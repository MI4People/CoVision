import { useEffect, useState } from "react";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import { TestArea } from "./getValidTestArea";
import * as tf from "@tensorflow/tfjs";

const MODEL_URL = "assets/classifier_model/model.json";
const modelPromise = loadGraphModel(MODEL_URL);

export enum TestResult {
  Negative,
  Unknown,
  Positive
}

function normalize(img: tf.Tensor4D, mean: [number, number, number], std: [number, number, number], axis: number) {
  const centeredRgb = new Array(img.shape[axis]).fill(0).map((idx) =>
    tf.gather(img, [idx], axis)
      .sub(tf.scalar(mean[idx]))
      .div(tf.scalar(std[idx])));
  return tf.concat(centeredRgb, axis)
}

const evaluate = (result: number | null) => {
  if (result == null) return TestResult.Unknown;
  if (result <= 0.2) return TestResult.Negative; // TODO adjust threshold
  if (result >= 0.8) return TestResult.Positive; // TODO adjust threshold
  return TestResult.Unknown;
}

const runAnalysis = async (testArea: TestArea): Promise<number | null> => {
  if (!testArea.input || !testArea.area) return null;
  const model = await modelPromise;
  const [width, height] = model.inputs[0].shape?.slice(2, 4) ?? [];
  const inputUnnormalized = tf.image.cropAndResize(
    testArea.input,
    tf.tensor2d([[
      testArea.area.left,
      testArea.area.top,
      testArea.area.right,
      testArea.area.bottom
    ]]).mul<tf.Tensor2D>(640),
    [0],
    [width, height],
  ).transpose<tf.Tensor4D>([0, 3, 1, 2]);

  const input = normalize(inputUnnormalized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1);
  const result_tf =
    (await model.executeAsync(input)) as tf.Tensor1D;

  const result = Array.from(result_tf.dataSync())[0];
  return result;
}

const useClassifierAnalysis = (testArea: TestArea) => {
  const [result, setResult] = useState<number | null>(null);
  useEffect(() => {
    runAnalysis(testArea).then(setResult);
  }, [testArea]);
  return [evaluate(result), result] as const;
}

export default useClassifierAnalysis;