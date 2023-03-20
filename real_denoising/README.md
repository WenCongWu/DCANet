### 1 dataset download

Download the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php), [RENOIR](https://ani.stat.fsu.edu/~abarbu/Renoir.html), and put these datasets into ./Datasets/ directory.

### 2 train

- Generate image patches from high-resolution training images of SIDD and RENOIR datasets
```
first: 
python generate_patches_SIDD.py --ps 180 --num_patches 200 --num_cores 16

second:
python generate_patches_RENOIR.py --ps 180 --num_patches 200 --num_cores 16
```

- train the model with default arguments by running

```
python train.py
```

### 3 test

- download the [DND](https://noise.visinf.tu-darmstadt.de/), and put these datasets into ./Datasets/ directory.

### testing on SIDD dataset

```
python test_SIDD.py
```

### testing on DND dataset

```
python test_DND.py
```

### in order to get the PSNR and SSIM values of the denoised SIDD test set, run MATLAB script
```
evaluate_SIDD.m
```
