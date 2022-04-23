import { MutableRefObject, useCallback, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import Webcam from "react-webcam";

const MODEL_URL = "assets/yolov5s_rapid_test_web_model/model.json";
const modelPromise = loadGraphModel(MODEL_URL);

export type Yolov5AnalysisResult = {
  boxes?: number[];
  scores?: number[];
  classes?: number[];
  valid_detections?: number;
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
    const [boxes_tf, scores_tf, classes_tf, valid_detections_tf] =
      (await model.executeAsync(input)) as tf.Tensor<tf.Rank>[];

    const boxes = Array.from(boxes_tf.dataSync());
    const scores = Array.from(scores_tf.dataSync());
    const classes = Array.from(classes_tf.dataSync());
    const valid_detections = valid_detections_tf.dataSync()[0];
    return { boxes, scores, classes, valid_detections }
  }, [])

  const [lastResult, setLastResult] = useState<Yolov5AnalysisResult | null>(null);
  useEverySecond(
    useCallback(async () => {
      if (!webcamRef.current) return;
      setLastResult(await runAnalysis(webcamRef.current));
    }, [runAnalysis, webcamRef])
  );
  return lastResult;
}

export default useYolov5Analysis;