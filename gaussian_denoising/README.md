### How to run the DCANet model for Gaussian denoising

### 1 dataset download

download the [DIV2K_HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), and put it into the /gaussian_denoising/trainsets/.

### 2 get image patches

### 2.1 get color image patches

```
        python utils/utils_image.py
```        

### 2.2 get grayscale image patches

```
        python convert_gray.py
```        
   
### 3. Train DCANet

```
python train.py 

Note: for the training of grayscale and color images, you need to modify the parameters of the gaussian_denoising/options/train_dcanet.json file, including n_channels, dataroot_H, in_nc and out_nc.
```

### 4. Test DCANet

```
python test.py
```
