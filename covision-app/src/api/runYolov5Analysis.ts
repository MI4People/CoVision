import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import Webcam from "react-webcam";
import { Rank } from "@tensorflow/tfjs";

const MODEL_URL = "assets/yolov5s_rapid_test_web_model/model.json";
const modelPromise = loadGraphModel(MODEL_URL);

export type Yolov5AnalysisResult = {
  input?: tf.Tensor<Rank.R4>,
  boxes?: number[];
  scores?: number[];
  classes?: number[];
  valid_detections?: number;
}

const runYolov5Analysis = async (webcam: Webcam): Promise<Yolov5AnalysisResult> => {
  const model = await modelPromise;
  const [width, height] = model.inputs[0].shape?.slice(1, 3) ?? [];
  const canvas = webcam.getCanvas({ width, height });
  if (!canvas) throw new Error('could not take screenshot');

  const input = tf.browser.fromPixels(canvas).div(255.0).expandDims<tf.Tensor4D>();
  const [boxes_tf, scores_tf, classes_tf, valid_detections_tf] =
    (await model.executeAsync(input)) as tf.Tensor<tf.Rank>[];

  const boxes = Array.from(boxes_tf.dataSync());
  const scores = Array.from(scores_tf.dataSync());
  const classes = Array.from(classes_tf.dataSync());
  const valid_detections = valid_detections_tf.dataSync()[0];
  return { input, boxes, scores, classes, valid_detections }
};

export default runYolov5Analysis;