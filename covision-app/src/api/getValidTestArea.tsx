import { Yolov5AnalysisResult } from './runYolov5Analysis';

export type TestArea = {
  score: number;
  area?: { left: number; right: number; top: number; bottom: number };
  input_tf?: Yolov5AnalysisResult['input_tf'];
};

export const getValidTestArea = ({ input_tf, scores, boxes }: Yolov5AnalysisResult): TestArea => {
  if (!input_tf || !scores || !boxes) return { score: -1 };
  // sort the scores by descendingly and pick the index of the highest score
  const bestScoreIdx = scores.map((s, idx) => [idx, s]).sort((a, b) => b[1] - a[1])[0][0];
  const score = scores[bestScoreIdx];

  const foundTest = score >= 0.6; // threshold
  if (!foundTest) return { score };

  const [left, top, right, bottom] = boxes.slice(bestScoreIdx * 4, bestScoreIdx * 4 + 4);
  return { input_tf, score, area: { left, top, right, bottom } };
};
