import { useCallback, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";

const MODEL_URL = "assets/yolov5s/model.json";

const modelPromise = loadGraphModel(MODEL_URL);

const useEverySecond = (callback: () => void) => {
  useEffect(() => {
    const handle = setInterval(callback, 1000);
    return () => clearInterval(handle);
  }, [callback]);
};

const CovCamera: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);

  useEverySecond(
    useCallback(async () => {
      const model = await modelPromise;
      const [width, height] = model.inputs[0].shape?.slice(1, 3) ?? [];
      const canvas = webcamRef.current?.getCanvas({ width, height });
      if (!canvas) return;

      const input = tf.browser.fromPixels(canvas).div(255.0).expandDims();
      const [boxes, scores, classes, valid_detections] =
        (await model.executeAsync(input)) as tf.Tensor<tf.Rank>[];

      const boxes_data = boxes.dataSync();
      const scores_data = scores.dataSync();
      const classes_data = classes.dataSync();
      const valid_detections_data = valid_detections.dataSync()[0];
      console.log({
        boxes_data,
        scores_data,
        classes_data,
        valid_detections_data,
      });
    }, [])
  );

  return (
    <Webcam
      ref={webcamRef}
      videoConstraints={{
        facingMode: "environment",
      }}
      width={1000}
      height={1000}
    />
  );
};

export default CovCamera;
