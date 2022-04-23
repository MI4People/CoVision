import * as tf from '@tensorflow/tfjs';

const warmUp = async (modelPromise: Promise<tf.GraphModel>) => {
  const model = await modelPromise;
  const inputShape = model.inputs[0].shape;
  if (!inputShape) return;
  const dummyInput = tf.zeros(inputShape);
  const res = await model.executeAsync(dummyInput);
  if (Array.isArray(res)) {
    res.forEach((r) => r.dispose());
  } else {
    res.dispose();
  }
  dummyInput.dispose();
};

export default warmUp;
