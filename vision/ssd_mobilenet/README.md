# Mobilenet v2 SSD with Coco 

This directory contains the Mobilenet v2 SSD object detetion models used for MLPerf. The following instrcutions are from https://github.com/mlperf/mobile_app/blob/model_repo_v0.7/tflite/README.md:

1.  ssd_mobilenet_v2_300_float

    *   Source:
        http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
        listed on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    *   After downloading, the frozen graph is exported using
        export_tflite_ssd_graph.py:

        ```
        <path to be configured>/export_tflite_ssd_graph.py \
        --pipeline_config_path=pipeline.config \
        --output_directory=<output path> \
        --trained_checkpoint_prefix=model.ckpt \
        --add_postprocessing_op=true --use_regular_nms=true
        ```

    *   Then it is converted to TFLite format using tflite_convert:

        ```
        tflite_convert --graph_def_file tflite_graph.pb \
        --enable_v1_converter \
        --experimental_new_converter=False \
        --output_file /tmp/ssdv2.tflite \
        --input_shapes=1,300,300,3 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
        --change_concat_input_ranges=false \
        --allow_custom_ops
        ```

2.  ssd_mobilenet_v2_300_uint8

    *   Source:
        http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
    *   After downloading, the frozen graph is converted to TFLite format using
        tflite_convert:

        ```
        tflite_convert --graph_def_file tflite_graph.pb \
        --enable_v1_converter \
        --experimental_new_converter=False \
        --output_file /tmp/ssdv2.tflite \
        --input_shapes=1,300,300,3 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
        --inference_type=QUANTIZED_UINT8 \
        --mean_values=128 --std_dev_values=128 \
        --change_concat_input_ranges=false \
        --allow_custom_ops
        ```

## Accuracy

|Precision   | Model file | Accuracy on CPU/GPU | 
|------------|-----------:|--------------------:|
|FP32        | TF         | 0.244               |
|FP32        | TFLite     | 0.245               |
|UINT8 (QAT) | TF         | 0.244               |
|UINT8 (QAT) | TFLite     | 0.229               |

Required accuracy to submit: 93% of FP32 TF model (above 0.227)

### Steps for evaluating accuracy on device
Follow instructions to build mlperf mobile app from vendor submission repo(s) and install on android device.
Download and prepare COCO dataset per instructions at https://github.com/mlcommons/mobile_app/blob/master/cpp/datasets/README.md.
Copy COCO validation set (complete dataset) to /sdcard/mlperf_datasets/coco/img/ and ground truth file (coco_val.pbtxt) to /sdcard/mlperf_datasets/coco/ (e.g., via adb)
Open mlperf mobile app on device and test detection benchmark. All validation images available at /sdcard/mlperf_datasets/coco/img/
Note: the evaluation script is at https://github.com/mlcommons/mobile_app/cpp/datasets/coco.cc
