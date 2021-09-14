# MobileDet SSD

This directory contains MobileDet SSD models generated via `export_tflite_ssd_graph.py` and `export_inference_graph.py`. 
*   Tensorflow source file: http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz
* `export TF_MODEL_DIR=/path/to/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19` (extracted from above)
### Generating TFLite-compatible files (export_tflite_ssd_graph.py)
1.  FP32  

    *   Edit the `score_threshold` to 0.3 in `${TF_MODEL_DIR}/fp32/pipeline.config`. Create folder `export_tflite_ssd_graph` under `${TF_MODEL_DIR}/fp32/`, and export the frozen graph using 
        `export_tflite_ssd_graph.py` with regular nms post-processing:

        ```
        python object_detection/export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path ${TF_MODEL_DIR}/fp32/pipeline.config --trained_checkpoint_prefix ${TF_MODEL_DIR}/fp32/model.ckpt-400000 --output_directory ${TF_MODEL_DIR}/fp32/export_tflite_ssd_graph/ --add_postprocessing_op=true --use_regular_nms=true --max_detections=10
        ```

    *   Then this `tflite_graph.pb` is converted to TFLite format using TOCO:

        ```
        bazel run -c opt tensorflow/lite/toco:toco -- --input_file=${TF_MODEL_DIR}/fp32/export_tflite_ssd_graph/tflite_graph.pb --output_file=${TF_MODEL_DIR}/fp32/export_tflite_ssd_graph/mobiledet.tflite --input_shapes=1,320,320,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=FLOAT --allow_custom_ops
        ```

2.  QAT

    *   Edit the `score_threshold` to 0.3 in `${TF_MODEL_DIR}/uint8/pipeline.config`. Create folder `export_tflite_ssd_graph` under `${TF_MODEL_DIR}/uint8/`, and export the frozen graph using 
        `export_tflite_ssd_graph.py`  with regular nms post-processing:

        ```
        python object_detection/export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path ${TF_MODEL_DIR}/uint8/pipeline.config --trained_checkpoint_prefix ${TF_MODEL_DIR}/uint8/model.ckpt-400000 --output_directory ${TF_MODEL_DIR}/uint8/export_tflite_ssd_graph/ --add_postprocessing_op=true --use_regular_nms=true --max_detections=10
        ```
    
    *   Then this `tflite_graph.pb` is converted to TFLite format using TOCO:

        ```
        bazel run -c opt tensorflow/lite/toco:toco -- --input_file=${TF_MODEL_DIR}/uint8/export_tflite_ssd_graph/tflite_graph.pb --output_file=${TF_MODEL_DIR}/uint8/export_tflite_ssd_graph/mobiledet_qat.tflite --input_shapes=1,320,320,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=QUANTIZED_UINT8 --allow_custom_ops --mean_values=128 --std_values=128
        ```

### Generating non-TFLite specific files (export_inference_graph.py)

For some reason `max_total_detections` cannot be lower than `max_detections_per_class`. In order to by-pass limitation, comment L86-88 in [object_detection/builders/post_processing_builder.py](https://github.com/tensorflow/models/blob/master/research/object_detection/builders/post_processing_builder.py#L86) (tried [master@9d1a69](https://github.com/tensorflow/models/tree/9d1a6927c6eb30312834dd7c63ad8a307c547b8d)).

1.  FP32

    *   Create folder `export_inference_graph` under `${TF_MODEL_DIR}/fp32/`. Run the following:

        ```
        python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=${TF_MODEL_DIR}/fp32/pipeline.config --config_override ' model{ ssd { post_processing { batch_non_max_suppression { score_threshold: 0.3 max_total_detections: 10 use_class_agnostic_nms: false max_detections_per_class: 100  }  }  }  }' --trained_checkpoint_prefix=${TF_MODEL_DIR}/fp32/model.ckpt-400000 --add_postprocessing_op=true --output_directory=${TF_MODEL_DIR}/fp32/export_inference_graph/
        ```

2. QAT

    *   Create folder `export_inference_graph` under `${TF_MODEL_DIR}/uint8/`. Run the following:

        ```
        python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=${TF_MODEL_DIR}/uint8/pipeline.config --config_override ' model{ ssd { post_processing { batch_non_max_suppression { score_threshold: 0.3 max_total_detections: 10 use_class_agnostic_nms: false max_detections_per_class: 100  }  }  }  }' --trained_checkpoint_prefix=${TF_MODEL_DIR}/uint8/model.ckpt-400000 --add_postprocessing_op=true --output_directory=${TF_MODEL_DIR}/uint8/export_inference_graph/
        ```

## Accuracy
The accuracy is evaluated on the server and with COCO 2017 validation images at 320x320 input resolution. 

The resulting tflite models generated via `export_tflite_ssd_graph.py` and TOCO have 0.285 and 0.278 mAP, for "fp32/export_tflite_ssd_graph/mobiledet.tflite" and "uint8/export_tflite_ssd_graph/mobiledet_qat.tflite", respectively.

The frozen graph models generated via `export_inference_graph.py` have 0.285 and 0.281 mAP for 
"fp32/export_inference_graph/frozen_inference_graph.pb" and "uint8/export_inference_graph/frozen_inference_graph.pb", respectively.


### Steps for evaluating accuracy on device
1. Follow instructions to build MLPerf mobile app from vendor submission repo(s) and install on android device.
2. Download and unzip the COCO 2017 validation set from http://images.cocodataset.org/zips/val2017.zip
3. Copy the images to /sdcard/mlperf_datasets/coco/img/ (e.g., using adb).
4. Open mlperf mobile app on device and test detection benchmark. Note: the evaluation script is at "https://github.com/mlcommons/mobile_app/blob/master/cpp/datasets/coco.cc". Also the normalized ground truth file is already part of the app at "https://github.com/mlcommons/mobile_app/blob/master/java/org/mlperf/inference/assets/coco_val.pbtxt" .

