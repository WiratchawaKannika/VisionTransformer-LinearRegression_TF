# VisionTransformer-LinearRegression_TF
Train Linear regression using VisionTransformer model with TensorFlow

## 1. Dependencies
* Ubuntu 16.04 or higher (64-bit)
* Cuda >= 11.4


## 2. Installation

```
$conda create -n vit-tf python=3.8 -y
$conda activate vit-tf
$pip install --user ipykernel
$python -m ipykernel install --user --name=vit-tf

***
$conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
$export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
$mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
$conda list | grep cuda

$pip install --upgrade pip
$pip install tensorflow-addons==0.16.1
$pip install tensorflow-gpu==2.4.0
$pip3 install --user tensorflow-gpu==2.11.0
$pip install vit-keras
$pip install opencv-python
$conda install scikit-learn -y
$pip install seaborn
$pip install numpy==1.20.3
```


### 3. Verify install
*Verify the GPU setup:
```
- import tensorflow as tf
- print(tf.__version__)
```

```
- physical_devices = tf.config.list_physical_devices('GPU') 
- print("Num GPUs:", len(physical_devices))
```

``` 
- print('Num GPUs Available:', len(tf.config.experimental.list_physical_devices('GPU')))
```

```
- tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

'2022-09-29 15:09:18.107884: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /device:GPU:0 with 9454 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:17:00.0, compute capability: 7.5
2022-09-29 15:09:18.108402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /device:GPU:1 with 9629 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:65:00.0, compute capability: 7.5
True'
```

```
- python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If a list of GPU devices is returned, you've installed TensorFlow successfully.

--------------------
