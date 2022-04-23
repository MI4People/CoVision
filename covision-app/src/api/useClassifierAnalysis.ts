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

const runAnalysis = async (testArea: TestArea): Promise<TestResult> => {
  if (!testArea.input || !testArea.area) return TestResult.Unknown;
  const model = await modelPromise;
  const [width, height] = model.inputs[0].shape?.slice(2, 4) ?? [];
  const input = tf.image.cropAndResize(
    testArea.input,
    tf.tensor2d([[
      testArea.area.left,
      testArea.area.top,
      testArea.area.right,
      testArea.area.bottom
    ]]).mul<tf.Tensor2D>(640),
    [0],
    [width, height],
  ).transpose([0, 3, 1, 2]);
  const result_tf =
    (await model.executeAsync(input)) as tf.Tensor<tf.Rank>;

  const result = Array.from(result_tf.dataSync())[0];
  if (result <= 0.2) return TestResult.Negative; // TODO adjust threshold
  if (result >= 0.8) return TestResult.Positive; // TODO adjust threshold
  return TestResult.Unknown;
}

const useClassifierAnalysis = (testArea: TestArea) => {
  const [result, setResult] = useState<TestResult>(TestResult.Unknown);
  useEffect(() => {
    runAnalysis(testArea).then(setResult);
  }, [testArea]);
  return result;
}

export default useClassifierAnalysis;