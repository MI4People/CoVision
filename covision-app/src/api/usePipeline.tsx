import { MutableRefObject, useCallback, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { getValidTestArea, TestArea } from './getValidTestArea';
import runClassifierAnalysis, { evaluate, TestResult } from './runClassifierAnalysis';
import runYolov5Analysis from './runYolov5Analysis';
import { BarcodeScanResult, runBarcodeScan } from './runBarcodeScan';

const useEvery = (ms: number, callback: () => Promise<void>) => {
  useEffect(() => {
    let isActive = true;
    const addTimeout = () => {
      setTimeout(async () => {
        try {
          await callback();
        } finally {
          if (isActive) addTimeout();
        }
      }, ms);
    };
    addTimeout();
    return () => {
      isActive = false;
    };
  }, [ms, callback]);
};

type AnalysisResult = {
  result: TestResult;
  detectionScore: number;
  classificationScore?: number | null;
  area?: TestArea['area'];
  barcodeResult?: BarcodeScanResult;
};

const usePipeline = (webcamRef: MutableRefObject<Webcam | null>) => {
  const [lastResult, setLastResult] = useState<AnalysisResult>({ result: TestResult.Unknown, detectionScore: -1 });
  useEvery(
    1000,
    useCallback(async () => {
      if (!webcamRef.current) return;

      const screenshot = webcamRef.current.getScreenshot({ width: 320, height: 640 });
      const barcodeTask = screenshot ? runBarcodeScan(screenshot) : undefined;

      const yolov5Res = await runYolov5Analysis(webcamRef.current);
      const testArea = getValidTestArea(yolov5Res);
      const classificationScore = await runClassifierAnalysis(testArea);

      yolov5Res.input_tf?.dispose();

      const result = evaluate(classificationScore);

      const barcodeResult = await barcodeTask;

      setLastResult({
        result,
        classificationScore,
        detectionScore: testArea.score,
        area: testArea.area,
        barcodeResult,
      });
    }, [webcamRef])
  );
  return lastResult;
};

export default usePipeline;
