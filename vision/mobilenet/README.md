# MobileNetEdge with ImageNet2012
## There are two folders in [models folder](models_and_code/checkpoints)
### [float](models_and_code/checkpoints/float) that has frozen pb files* with EMA enabled for TF-FP32, TF-Lite INT8, TF-Lite UNIT8 (* Contributor - Suyog Gupta from Google)
### [mobilenet_edgetpu_224_1.0](models_and_code/checkpoints/mobilenet_edgetpu_224_1.0) that has the intial check points from Google for MobileNetEdge


## Accuracy
|Model | Tensorflow 2 | Tensorflow 1.x|Tensorflow-lite | 
|------|------------------:|-----------------:|-----------------:|
|FP32  | [76.19](models_and_code/checkpoints/float/edge_frozen_graph.pb)           | [76.19](models_and_code/checkpoints/float/frozen_graph_tf1x_transform.pb)          |[75.94](models_and_code/checkpoints/float/edge_float_dm1p0.tflite)          |
|INT8 (PTQ)  | N/A          | N/A          | ??          |
||||

### Steps for evaluating accuracy on device

1. Follow instructions to build and install mlperf mobile app from commit c8075 on android device
               https://github.com/mlcommons/mobile_app/tree/c8075ac367554cae98b1508c4a5ab14d14c5885c
2. Download imagenet per instructions at 
               https://github.com/mlcommons/mobile_app/tree/c8075ac367554cae98b1508c4a5ab14d14c5885c/cpp/datasets
3. Copy imagenet validation set and ground truth file to /sdcard/img_val (e.g., via adb)
4. Open mlperf mobile app on device and test classification benchmark. All images available at /sdcard/img_val 

Note: the evaluation script is at https://github.com/mlcommons/mobile_app/blob/c8075ac367554cae98b1508c4a5ab14d14c5885c/cpp/datasets/imagenet.cc .

