## Learning with Feature-Dependent Label Noise: A Progressive Approach (ICLR 2021, spotlight) [Paper](https://openreview.net/pdf?id=ZPa2SyGcbwh)

![](https://github.com/pxiangwu/PLC/blob/master/teaser.png)

## Requirements
- PyTorch 1.6 (Other versions >= 1.0 should also work)
- Python 3.8.5 (Other Python 3.x should also work)
- tqdm, termcolor, etc (which can be easily installed via pip)

## Usage
- The folder `cifar` contains the code for generating PMD noise and running on synthetic datasets.
- The folder `clothing1m` contains the code for running on Clothing1M dataset.
- The folder `Food101` contains the code for running on Food-101N dataset.

## Reference
```
@inproceedings{prog_noise_iclr2021,
  title={Learning with Feature-Dependent Label Noise: A Progressive Approach},
  author={Zhang, Yikai and Zheng, Songzhu and Wu, Pengxiang and Goswami, Mayank and Chen, Chao},
  booktitle={ICLR},
  year={2021}
}
```
## Related Work
- Error-Bounded Correction of Noisy Labels. In *ICML*, 2020. [[Paper]](https://arxiv.org/pdf/2011.10077.pdf)[[Code]](https://github.com/pingqingsheng/LRT)
- A Topological Filter for Learning with Label Noise. In *NeurIPS*, 2020. [[Paper]](https://proceedings.neurips.cc/paper/2020/file/f4e3ce3e7b581ff32e40968298ba013d-Paper.pdf)[[Code]](https://github.com/pxiangwu/TopoFilter)
