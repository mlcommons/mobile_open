##  Prepping Datasets for Accuracy Measurement on MLPerf Mobile App
### ImageNet
Download ImageNet per instructions at https://github.com/mlcommons/mobile_app/tree/c8075ac367554cae98b1508c4a5ab14d14c5885c/cpp/datasets#imagenet

Copy validation set images to `/sdcard/mlperf_datasets/imagenet/img`. The ground truth file is already part of the mobile app. 

### COCO
Download COCO 2017 validation dataset from http://images.cocodataset.org/zips/val2017.zip

Copy validation set images (.jpg) to `/sdcard/mlperf_datasets/coco/img`. The ground truth file is already part of the mobile app. 

### ADE20K 
Follow the instructions at https://github.com/mlcommons/mobile/blob/prep-dataset-accuracy/vision/deeplab/ade20k_512x512_dataset_on_device/README.md

You should be copying the validation images (.jpg files) to `/sdcard/mlperf_datasets/ade20k/images` and the associated ground truth (.raw) files to `/sdcard/mlperf_datasets/ade20k/annotations`.

### SQUAD

This dataset (v1.1/dev) is already part of the mobile app.
