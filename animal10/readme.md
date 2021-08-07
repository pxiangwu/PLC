Suppose we are in the folder `animal10`.

## Data Preparation
Download ANIMAL-10N ([Link](https://dm.kaist.ac.kr/datasets/animal-10n/)).

Specify the data path in file `animal10/dataset.py`:

```
if data_path is None:
    data_path = '/home/pwu/Downloads/animal10'
```

## Training
Run the command:
```
python train.py
```
