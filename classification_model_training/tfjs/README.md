
```bash
npm install @tensorflow/tfjs-node
npm install @tensorflow/tfjs-node-gpu
```

```bash
./eval.js \
    PATH_TO_DATASET/test \
    PATH_TO_MODEL/model.json \
    --gpu \
    --normalize ImagenetMeanStd
```
