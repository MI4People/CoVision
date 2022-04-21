import { MutableRefObject, useCallback, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import Webcam from "react-webcam";

const MODEL_URL = "assets/yolov5s_rapid_test_web_model/model.json";
const modelPromise = loadGraphModel(MODEL_URL);

type Yolov5AnalysisResult = {
  boxes_data: Float32Array;
  scores_data: Float32Array;
  classes_data: Float32Array;
  valid_detections_data: number;
}

const useEverySecond = (callback: () => void) => {
  useEffect(() => {
    const handle = setInterval(callback, 1000);
    return () => clearInterval(handle);
  }, [callback]);
};

const useYolov5Analysis = (webcamRef: MutableRefObject<Webcam | null>) => {
  const runAnalysis = useCallback(async (webcam: Webcam): Promise<Yolov5AnalysisResult | null> => {
    const model = await modelPromise;
    const [width, height] = model.inputs[0].shape?.slice(1, 3) ?? [];
    const canvas = webcam.getCanvas({ width, height });
    if (!canvas) return null;

    const input = tf.browser.fromPixels(canvas).div(255.0).expandDims();
    const [boxes, scores, classes, valid_detections] =
      (await model.executeAsync(input)) as tf.Tensor<tf.Rank>[];

    const boxes_data = boxes.dataSync() as Float32Array;
    const scores_data = scores.dataSync() as Float32Array;
    const classes_data = classes.dataSync() as Float32Array;
    const valid_detections_data = valid_detections.dataSync()[0];
    return { boxes_data, scores_data, classes_data, valid_detections_data }
  }, [])

  const [lastResult, setLastResult] = useState<Yolov5AnalysisResult|null>(null);
  useEverySecond(
    useCallback(async () => {
      if (!webcamRef.current) return;
      setLastResult(await runAnalysis(webcamRef.current));
    }, [runAnalysis, webcamRef])
  );
  return lastResult;
}

export default useYolov5Analysis;