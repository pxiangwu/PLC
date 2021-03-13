Suppose we are in the folder `cifar`.

## Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in our experiments are provided in folder `noisy_labels`.

If you want to generate the PMD noisy labels by yourself, run the command like below:
```
python gen_noise_label.py --dataset cifar10 --noise_factor 1.2 --epochs 25 --noise_type 1 --gpus 0
```

The above command will generate a `cifar10-1-0.35.npy` file into folder `noisy_labels`.
For the name `cifar10-1-0.35.npy`, cifar10 means we are generating label noise for cifar10 dataset; the 1 in the middle means we are using type-I PMD noise; the 0.35 means the real noise level is 35%.
- `dataset`: two options [cifar10 | cifar100].
- `epochs`: the number of training epochs.
- `noise_factor`: it is used to control the real noise level. As is mentioned in our paper, "To change the noise level, we multiply \tau by a certain constant factor such that the final proportion of noise matches our requirement."

On Ubuntu 18.04.4, with Python 3.8.5 and PyTorch 1.6, we use the following parameters for noisy label generation. 
For cifar10,
(1) type-I noise:
- Setting `noise_factor=1.43` and `epochs = 25` gives real noise level 0.35
- Setting `noise_factor=3.40` and `epochs = 25` gives real noise level 0.70

(2) type-II noise:
- Setting `noise_factor=1.32` and `epochs = 25` gives real noise level 0.35
- Setting `noise_factor=3.10` and `epochs = 25` gives real noise level 0.70

(3) type-III noise:
- Setting `noise_factor=1.35` and `epochs = 25` gives real noise level 0.35
- Setting `noise_factor=3.20` and `epochs = 25` gives real noise level 0.70

For cifar100,
(1) type-I noise:
- Setting `noise_factor=1.08` and `epochs = 25` gives real noise level 0.35
- Setting `noise_factor=1.80` and `epochs = 10` gives real noise level 0.70

(2) type-II noise:
- Setting `noise_factor=1.02` and `epochs = 35` gives real noise level 0.35
- Setting `noise_factor=2.58` and `epochs = 35` gives real noise level 0.70

(3) type-III noise:
- Setting `noise_factor=1.12` and `epochs = 35` gives real noise level 0.35
- Setting `noise_factor=1.20` and `epochs = 10` gives real noise level 0.70


## Training
For PMD noise, run the command:
```
python train.py --noise_label_file cifar10-1-0.35.npy --noise_type uniform --noise_level 0 --delta 0.3 --inc 0.1 --nepoch 180 --warm_up 8 --gpus 0 --seed 77
python train.py --noise_label_file cifar100-1-0.35.npy --noise_type uniform --noise_level 0 --delta 0.3 --inc 0.1 --nepoch 180 --warm_up 8 --gpus 0 --seed 77
```
For hybrid noise, run the command:
```
python train.py --noise_label_file cifar10-1-0.35.npy --noise_type uniform --noise_level 0.3 --delta 0.3 --inc 0.1 --nepoch 180 --warm_up 8 --gpus 0 --seed 77
python train.py --noise_label_file cifar100-1-0.35.npy --noise_type uniform --noise_level 0.3 --delta 0.3 --inc 0.1 --nepoch 180 --warm_up 15 --gpus 0 --seed 77
```
Some major parameters:
- `noise_label_file`: the `.npy` file for PMD noise labels.
- `noise_type`: the i.i.d. noise type (`uniform` or `asymmetric`).
- `noise_level`: the noise level for i.i.d. noise. If `noise_level = 0`, then only apply the PMD noise. Otherwise, apply the hybrid noise which contains PMD and i.i.d. noise.
- `delta`: the initial threshold, i.e., the T0 in Algorithm 1.
- `inc`: the step size for the adjustment of threshod, i.e., the beta in Algorithm 1.
- `nepoch`: the number of training epochs.
- `warm_up`: the number of warm_up epochs.
- `seed`: the random seed.
