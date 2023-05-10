# how to generate the dataset:
as said in https://github.com/mlcommons/mobile_app_open/blob/master/android/cpp/datasets/README.md, we can use
a simple Python script to generate 512x512 images and ground truth files using DeepLab
preprocessing parts.

To use jpeg files as inputs:

```python
import os
import tensorflow as tf
import deeplab.input_preprocess
from PIL import Image as Image

tf.enable_eager_execution()

home = os.getenv("HOME")
ADE20K_PATH = home + '/tf-models/research/deeplab/datasets/ADE20K/ADEChallengeData2016/'

for i in range(1, 2001):
    image_jpeg = ADE20K_PATH+f'images/validation/ADE_val_0000{i:04}.jpg'
    label_png = ADE20K_PATH+f'annotations/validation/ADE_val_0000{i:04}.png'

    image_jpeg_data = tf.io.read_file(image_jpeg)
    image_tensor = tf.io.decode_jpeg(image_jpeg_data)
    label_png_data = tf.io.read_file(label_png)
    label_tensor = tf.io.decode_png(label_png_data)
    o_image, p_image, p_label = deeplab.input_preprocess.preprocess_image_and_label(image_tensor, label_tensor, 512, 512, 512, 512, is_training=False)

    target_image_jpeg = f'/tmp/ade20k_512/images/validation/ADE_val_0000{i:04}.jpg'
    target_label_png = f'/tmp/ade20k_512/annotations/ADE_val_0000{i:04}.png'

    resized_image = Image.fromarray(tf.reshape(tf.cast(p_image, tf.uint8), [512, 512, 3]).numpy())
    resized_image.save(target_image_jpeg, quality=100, subsampling=0))

    resized_label = Image.fromarray(tf.reshape(tf.cast(p_label, tf.uint8), [512, 512]).numpy())
    resized_label.save(target_label_png)
```

To use PNG files as inputs:
```python

home = os.getenv("HOME")
ADE20K_PATH = home + '/tf-models/research/deeplab/datasets/ADE20K/ADEChallengeData2016/'

for i in range(1, 2001):
    image_jpeg = ADE20K_PATH+f'images/validation/ADE_val_0000{i:04}.jpg'
    label_png = ADE20K_PATH+f'annotations/validation/ADE_val_0000{i:04}.png'

    image_jpeg_data = tf.io.read_file(image_jpeg)
    image_tensor = tf.io.decode_jpeg(image_jpeg_data)
    label_png_data = tf.io.read_file(label_png)
    label_tensor = tf.io.decode_png(label_png_data)
    o_image, p_image, p_label = deeplab.input_preprocess.preprocess_image_and_label(image_tensor, label_tensor, 512, 512, 512, 512, is_training=False)

    target_image_png = f'/tmp/ade20k_512/images/validation/ADE_val_0000{i:04}.png'
    target_label_png = f'/tmp/ade20k_512/annotations/ADE_val_0000{i:04}.png'

    resized_image = Image.fromarray(tf.reshape(tf.cast(p_image, tf.uint8), [512, 512, 3]).numpy())
    resized_image.save(target_image_png)

    resized_label = Image.fromarray(tf.reshape(tf.cast(p_label, tf.uint8), [512, 512]).numpy())
    resized_label.save(target_label_png)
```

Here we assume that
1. you have cloned `https://github.com/tensorflow/models/` to `$HOME/tf-models/` and
prepared the ade20k dataset using deeplab scripts
2. you have `$HOME/tf-models/research` and `$HOME/tf-models/research/slim` in you PYTHONPATH

# how to use them
simply put the `/tmp/ade20k` directory into the place the mobile app looks for will work, e.g.,
```
cd /tmp
adb shell mkdir -p /sdcard/mlperf_datasets/
adb push ade20k /sdcard/mlperf_datasets/
```
