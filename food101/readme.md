Suppose we are in the folder `food101`.

## Data Preparation
Download Food-101N ([Link](https://kuanghuei.github.io/Food-101N/)) and Food-101 ([Link](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)).
Run the following command to prepare train/test labels (don't forget to modify the file paths in the code):
```
python dataset.py
```
The above command will generate `train_images.npy` and `test_images.npy`

## Training
Run the command:
```
python train.py --gpus 0 --warm_up 4 --rollWindow 4 --batch 32 --epoch 30 --lr 0.005 --seed 11 --delta 0.1
```
