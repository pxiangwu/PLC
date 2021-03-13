Suppose we are in the folder `clothing1m`.

## Data Preparation
Download the Clothing1M dataset. Then run the following command to prepare train/val/test labels (don't forget to modify the file paths in the code):
```
python data_clothing1m.py
```

## Training
Run the command:
```
python train_clothing1m.py --nepochs 15 --gpus 0,1,2,3,4,5,6,7 --warm_up 1 --top_k 3
```