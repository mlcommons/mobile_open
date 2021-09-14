# MLPerf - Image Segmentation - Deeplabv3

A Tensorflow implementation of Deeplabv3.\
This implementation focus on ADE20K dataset with 32 classes. The model architecture is fixed on Deeplabv3 with mobilenetv2 as backbone and output stride 16.\
Some of the source code is directly modified from the implementation under [tensorflow/models/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Content

1. [Environment](#environment)
2. [Data Generation](#data-generation)
3. [Train](#train)
4. [Evaluation](#evaluation)
5. [Quantization-aware training](#quantization-aware-training)
6. [Post-training quantization](#post-training-quantization)
7. [Reference](#reference)

## Environment

- Tensorflow 1.15 for training and ckpt, pb evaluation
- Tensorflow 2.2 for post-training quantization and tflite evaluation
- python3.6

## Data Generation

### Download ADE20K

```bsh
./download_and_convert_ade20k.sh
```

### Create TFRecord for ADE20K

```bash
python build_ade20k_data.py \
  --train_image_folder=${TRAIN_IMAGE_FOLDER} \
  --train_image_label_folder=${TRAIN_LABEL_FOLDER} \
  --val_image_folder=${VAL_IMAGE_FOLDER} \
  --val_image_label_folder=${VAL_LABEL_FOLDER} \
  --output_dir=${OUTPUT_DIR}
```

- TRAIN_IMAGE_FOLDER: The folder with all training images.
- TRAIN_LABEL_FOLDER:The folder with all training labels.
- VAL_IMAGE_FOLDER: The folder with all validation images.
- VAL_LABEL_FOLDER: The folder with all validation labels.
- OUTPUT_DIR: The folder where the output tfrecord will be.

## Train

Finetune on pretrained model with 32 classes, size 512x512, output stride 16.

```bash
python train.py \
    --logtostderr \
    --training_number_of_steps=200000 \
    --train_split="train" \
    --output_stride=16\
    --decoder_output_stride=4 \
    --train_crop_size="512,512" \
    --train_batch_size=4 \
    --min_resize_value=512 \
    --max_resize_value=512 \
    --resize_factor=16\
    --dataset="ade20k" \
    --tf_initial_checkpoint=${PRETRAINED_CKPT_FILE} \
    --train_logdir=${OUTPUT_DIR} \
    --dataset_dir=${DATA_DIR} \
    --fine_tune_batch_norm=False \
    --optimizer=adam \
    --adam_learning_rate=.000005 \
    --save_interval_secs=600 \
    --initialize_last_layer=False
```

- PRETRAINED_CKPT_FILE: A pretrained ckpt for finetuning. Here we used the [model](http://download.tensorflow.org/models/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz) downloaded from the official repo.
- DATA_DIR: Dataset folder, where the tfrecord is.
- OUTPUT_DIR: The folder where the output .ckpt will be stored.

## Evaluation

### Evaluate ckpt

To use official evaluation file, please use `eval.py` which will evaluate the result in the folder non-stop.

```bash
python eval.py \
    --checkpoint_dir=${CKPT_DIR} \
    --dataset_dir=${DATA_DIR} \
    --eval_logdir=log \
    --eval_crop_size="512,512" \
    --min_resize_value=512 \
    --max_resize_value=512 \
    --decoder_output_stride=4 \
    --resize_factor=16 \
    --output_stride=16
```

- CKPT_DIR: The folder where your ckpt is.
- DATA_DIR: Dataset folder, where the tfrecord is.

### Evaluate  GraphDef (pb)

#### Export pb from ckpt

```bash
python export_model.py \
    --checkpoint_path=${CKPT_DIR} \
    --export_path=${PB_OUTPUT_DIR} \
    --num_classes=32 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="512,512" \
    --min_resize_value=512 \
    --max_resize_value=512 \
    --resize_factor=16
```

- CKPT_DIR: The folder where your ckpt is.
- PB_OUTPUT_DIR: The folder where output pb will be.

#### Evaluate pb

```bash
python eval_pb.py \
    --pb=${PB_FILE} \
    --dataset_dir=${DATA_DIR} \
    --eval_logdir=log \
    --eval_crop_size="512,512" \
    --min_resize_value=512 \
    --max_resize_value=512 \
    --resize_factor=16

```

- PB_FILE: The path of your pb file.
- DATA_DIR: Dataset folder, where the tfrecord is.

### Evaluate TFLite

#### Generate data for TFLite evaluation

```bash
python gen_preprocessed_data.py \
  --eval_crop_size='512,512' \
  --min_resize_value=512 \
  --max_resize_value=512 \
  --dataset_dir=${DATA_DIR} \
  --preprocessed_dataset_dir=${OUTPUT_DIR}\
  --eval_split=val
```

- DATA_DIR: Dataset folder, where the tfrecord is.
- OUTPUT_DIR: Dataset folder, where the preprocessed data will be stored. A `images` and `annotations` folder will be created under this folder.

#### Run TFLite evaluation

```bash
python segmentation_inference_eval_seg_tflite.py
```

Note that please modify the index in the code.

- test_dir: Input data folder containing input image.
- val_label_dir: Annotation data folder containing annotation image.
- net_file: Path to TFLite file to be evaluate.
- is_float_input: A bool indicates input type is float or uint8.
- is_input_0_255: A bool indicates input of the model is [0, 255] or [-1, 1]. This depends on if the preprocessing is at the beginning of the model or not.

- **Note**: To evaluate our model:

  - `checkpoints/float/freeze.tflite`
    - IS_FLOAT = True
    - IS_INPUT_0_255 = False
  - `checkpoints/post_train_quant/freeze_ptq.tflite`
    - IS_FLOAT = False
    - IS_INPUT_0_255 = True
  - `checkpoints/quantize_aware_training/freeze_qat.tflite`
    - IS_FLOAT = False
    - IS_INPUT_0_255 = True

## Quantization-aware training

```bash
python train.py \
    --logtostderr \
    --training_number_of_steps=100000 \
    --train_split="train" \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="512,512" \
    --train_batch_size=4 \
    --min_resize_value=512 \
    --max_resize_value=512 \
    --resize_factor=16 \
    --dataset="ade20k" \
    --tf_initial_checkpoint=${PRETRAINED_CKPT_FILE} \
    --train_logdir=${OUTPUT_DIR} \
    --dataset_dir=${DATA_DIR} \
    --fine_tune_batch_norm=True \
    --base_learning_rate=.000001 \
    --save_interval_secs=600 \
    --quantize_delay_step=0
```

- PRETRAINED_CKPT_FILE: A pretrained ckpt for finetuning.
- DATA_DIR: Dataset folder, where the tfrecord is.
- OUTPUT_DIR: The folder where the output .ckpt will be stored.

- **Note 1**: To evaluate .ckpt or export to graphdef (.pb), one can use `eval.py` and `export_model.py` with additional flag `--quantize_delay_step=0`.
- **Note 2**: Although `fine_tune_batch_norm` is set to True, Batch-norms will be freezed immediately at step 0 due to setting argument `freeze_bn_delay=quantize_delay_step` in the code . This prevent updating batch-norm argument with small batches.

## Post-training quantization

### Generate data

#### Generate data for post-training quantization

1. Generate tfrecord with calibration data

    You may find 500 calibration images listed in `checkpoints/post_train_quant/ptq.txt`.\
    To download ADE20K, please refer to [Download ADE20K](#download-ade20k).\
    To convert the dataset to tfrecord, please refer to [Create TFRecord for ADE20K](#create-tfrecord-for-ade20k).

2. Gen Preprocessed data for quantization.

    ```bash
    python gen_preprocessed_data.py \
      --eval_crop_size='512,512' \
      --min_resize_value=512 \
      --max_resize_value=512 \
      --dataset_dir=${DATA_DIR} \
      --preprocessed_dataset_dir=${OUTPUT_DIR}\
      --eval_split=train
    ```

    - DATA_DIR: Dataset folder. Exactly the `TRAIN_IMAGE_FOLDER` in previous step.
    - OUTPUT_DIR: Dataset folder, where the preprocessed data will be stored. A `images` and `annotations` folder will be created under this folder.

### Run post-training quantization

```bash
python post_training_quant.py \
    --dataset_dir=${PREPROCESSED_DATA_DIR} \
    --graph_def=${INPUT_PB_FILE} \
    --out_tflite=${OUTPUT_TFLITE_FILE} \
    --input_type=uint8 \
    --sample_num=500
```

- PREPROCESSED_DATA_DIR: Dataset folder. Where the preprocessed input images are.
- INPUT_PB_FILE: The path of your Graphdef (.pb) to be quantized.
- OUTPUT_TFLITE_FILE: The path of your output tflite file.

## Reference

- Deeplab tensorflow implementation: [DeepLab: Deep Labelling for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab)
