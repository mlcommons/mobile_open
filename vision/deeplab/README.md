# DeepLabV3 with ADE20K
## directories
* [models and related code](models_and_code)
* [how to generate 512x512 ade20k images and ground truth files for on-device evaluation](ade20k_512x512_dataset_on_device)

## Accuracy
|model | checkpoint on x86 | frozen pb on x86 | tflite on x86 | tflite on device CPU | miou file|
|------|------------------:|-----------------:|--------------:|---------------------:|----------:|
|fp32  | 0.5528            | 0.5528           | N/A           | 0.5509               |[here](models_and_code/checkpoints/float/miou.txt)|
|QAT   | 0.5553            | 0.5553           | 0.5483        | 0.5467               |[here](models_and_code/checkpoints/quantize_aware_training/miou.txt)|
|PTQ   | N/A               | N/A              | 0.5469        | 0.5412               |[here](models_and_code/checkpoints/post_train_quant/miou.txt)|

### accuracy target
* back of the envolop calculation:
  * 0.5412 / 0.5509 ~= 0.9824
  * 0.54 / 0.5509 ~= 0.9802
  * 0.535 / 0.5509 ~= 0.9711

So, if mIoU >= 0.535, then set the accuracy target to be 97% should be safe.

## Evaluate accuracy on Android devices
1. prepare 512x512 images and groundtruth files as described in [how to generate 512x512 ade20k images and ground truth files for on-device evaluation](ade20k_512x512_dataset_on_device)
2. build command line program for android devices, e.g.,
```bash
$ bazel build -c opt \
  --config android_arm64 \
  --cxxopt='--std=c++14' \ 
  --host_cxxopt='--std=c++14' \
  //cpp/binary:main
```
3. install the binary, e.g.,
```
$ adb push bazel-bin/cpp/binary/main /data/local/tmp/mlperf_main
```
4. assuming we have what we prepared in 1. in `/sdcard/mlperf_datasets/ad20k`, run it
```
$ adb shell mkdir -p /data/local/tmp/ade20k_output
$ adb shell /data/local/tmp/mlperf_main tflite ade20k \
  --mode=SubmissionRun \
  --output_dir=/data/local/tmp/ade20k_output \
  --model_file=/data/local/tmp/mlperf_tflite/deeplabv3_mnv2_ade20k_uint8.tflite \
  --images_directory=/sdcard/mlperf_datasets/ade20k/images  \
  --ground_truth_directory=/sdcard/mlperf_datasets/ade20k/annotations  \
  --num_threads=4
$ adb pull /data/local/tmp/ade20k_output /tmp/
```
5. you should have
```
mlperf_log_accuracy.json	mlperf_log_detail.txt		mlperf_log_summary.txt		mlperf_log_trace.json
```
in `/tmp/ade20k_output/` and get output like the following:
```
native : main.cc:118 Using TFLite backend
INFO: Initialized TensorFlow Lite runtime.
native : tflite_inference_stage.cc:107 
native : tflite.cc:85 Applying delegate: none
native : tflite_inference_stage.cc:62 Tried to apply null TfLiteDelegatePtr to TfliteInferenceStage
native : main.cc:236 Using ADE20K dataset

No warnings encountered during test.

No errors encountered during test.
================================================
MLPerf Results Summary
================================================
SUT name : TFLite
Scenario : Single Stream
Mode     : Performance
90th percentile latency (ns) : 141426994
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes

================================================
Additional Stats
================================================
QPS w/ loadgen overhead         : 7.72
QPS w/o loadgen overhead        : 7.73

Min latency (ns)                : 121304127
Max latency (ns)                : 206884343
Mean latency (ns)               : 129386553
50.00 percentile latency (ns)   : 124973763
90.00 percentile latency (ns)   : 141426994
95.00 percentile latency (ns)   : 144560431
97.00 percentile latency (ns)   : 152314807
99.00 percentile latency (ns)   : 206884343
99.90 percentile latency (ns)   : 206884343

================================================
Test Parameters Used
================================================
samples_per_query : 1
target_qps : 1000
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 100
max_duration (ms): 0
min_query_count : 100
max_query_count : 0
qsl_rng_seed : 0
sample_index_rng_seed : 0
schedule_rng_seed : 0
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
print_timestamps : false
performance_issue_unique : false
performance_issue_same : false
performance_issue_same_index : 0
performance_sample_count : 635

No warnings encountered during test.

No errors encountered during test.
native : main.cc:288 90 percentile latency: 141.43 ms
native : main.cc:289 Accuracy: 0.5468 mIoU
```
