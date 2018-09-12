# MobileNet-PyTorch
This is pytorch implemention of mobile architecture，converted from [gluon model_zoo](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/mobilenet.py)

**Note**：

- The architecture is coming from paper, so the original image size is 224x224 and with rescale=32. however, here use cifar10 as training dataset (image size is 32x32, and resize to 64x64), so the accuracy in this data is not well. (just as demo, so I did not try to modify the architecture)
- The default dataset root in  `your_computer_name/data`

### 1. MobileNet v1

change  model name：

```python
# choose network --- choose 0
model_name = ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2'][0]
```

accuracy after epoch80：89.48%

### 2. MobileNet v2

change  model name：

```python
# choose network --- choose 1
model_name = ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2'][1]
```

accuracy after epoch80：89.29%

### 3. ShuffleNet v1

change  model name：

```python
# choose network --- choose 2
model_name = ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2'][2]
```

accuracy after epoch80：

### 4. ShuffleNet v2

change  model name：

```python
# choose network --- choose 3
model_name = ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2'][3]
```

accuracy after epoch80：

## Reference

1. [gluon model_zoo](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/mobilenet.py)：mobilenetv1&v2
2. [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py)：shufflenet（nearly all the code is copy）