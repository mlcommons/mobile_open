# MobileNetV4-Conv-Large with ImageNet2012

This folder containes the int8 TFLite of MobileNetV4-Conv-Large.
The TensorFlow Saved Model, FP32 TFLite files are uploaded in a seperate release (Link pending).
These models are used for the MLPerf Mobile Image Classification Task.

## Accuracy
|Model | FP32 | Int8 (PTQ) |
|------|------------------:|-----------------:|
|MobileNetV4-Conv-Large  | 82.68          | 81.79         |
||||

### Steps for evaluating accuracy on device

1. Follow instructions to build and install mlperf mobile app on android device
               https://github.com/mlcommons/mobile_app_open/blob/master/android/README.md
2. Download imagenet per instructions at 
               https://github.com/mlcommons/mobile_app_open/blob/master/android/cpp/datasets/README.md
3. Copy imagenet validation set and ground truth file to device (e.g., via adb)
4. Open mlperf mobile app on device and test classification benchmark.

Note: the evaluation script is at https://github.com/mlcommons/mobile_app_open/blob/master/android/cpp/datasets/imagenet.cc.

