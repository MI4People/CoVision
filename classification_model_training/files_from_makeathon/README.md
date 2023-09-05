#### train
```bash
python train.py --num_classes 4 --pretrained_on_ImageNet --fold 1 \
    --dataset DATASET_PATH/images \
    --gt DATASET_PATH/gt.csv \
    --outdir OUTPUT_PATH \
    --epochs 100 --seed 2
```

#### predict
```bash
python predict.py --num_classes 4 \
    --dataset PATH/images \
    --gt PATH/gt.csv \
    --single_model_path PATH/modelXXXX.pt
```

#### captum insights
```bash
python captum_insights.py --num_classes 4 \
    --dataset PATH/images \
    --gt PATH/gt.csv \
    --single_model_path PATH/modelXXX.pt
```