# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""
import pathlib

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from tensorflow.contrib import training as contrib_training
import common
import model
from datasets import data_generator
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS


# Settings for evaluating the model.
flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('eval_crop_size', '512,512',
                  'Image crop size [height, width] for evaluation.')

# Dataset settings.

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')
flags.DEFINE_string('preprocessed_dataset_dir', None, 'Where the preprocessed dataset be stored.')

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(
      split_name=FLAGS.eval_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.eval_batch_size,
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=1,
      num_readers=2,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])
    sess = tf.Session()
    sess.run(
        tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
    )

    img_dir = pathlib.Path(FLAGS.preprocessed_dataset_dir) / 'images'
    img_dir.mkdir(exist_ok=True, parents=True)

    anno_dir = pathlib.Path(FLAGS.preprocessed_dataset_dir) / 'annotations'
    anno_dir.mkdir(exist_ok=True, parents=True)

    step = 0
    while True:
        try:
            output_in_file = f'img_{step}.png'
            inp, label = sess.run([samples[common.IMAGE], samples[common.LABEL]])

            cv2.imwrite(str(img_dir / output_in_file), inp[0,...,::-1].astype(np.uint8))
            cv2.imwrite(str(anno_dir / output_in_file), label[0,:,:,::-1].astype(np.uint8))

            step +=1
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
