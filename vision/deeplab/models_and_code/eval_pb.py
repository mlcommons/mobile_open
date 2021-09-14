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

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from tensorflow.contrib import training as contrib_training
from tensorflow.python.tools import optimize_for_inference_lib
import common
import model
from datasets import data_generator

flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

#flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

flags.DEFINE_string('pb', None, 'GraphDef (pb) file name.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('eval_crop_size', '512,512',
                  'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


INPUT_NODE_NAME = 'MobilenetV2/MobilenetV2/input'
OUTPUT_NODE_NAME = 'ArgMax'

def import_pb(pb_file_name, inputs, input_node_name, output_node_name):
    """Import pb file.

    Args:
        pb_file_name: A string indicates the pb file name.
        inputs: A tensor indicates the model inputs.
        input_node_name: A string indicates the input node name.
        output_node_name: A string indicates the output node name.

    Returns
        A tensor indicates the output of the model.
    """
    with tf.io.gfile.GFile(pb_file_name, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # check basic properties for the graph
    optimize_for_inference_lib.ensure_graph_is_valid(graph_def)

    input_map = {input_node_name: inputs}
    tf.import_graph_def(graph_def, input_map=input_map, name='')

    return tf.compat.v1.get_default_graph().get_tensor_by_name(output_node_name + ':0')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(
      split_name=FLAGS.eval_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.eval_batch_size,
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      num_readers=2,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
        output_stride=FLAGS.output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])

    # preprocessing normalize
    samples[common.IMAGE] = (2.0 / 255.0) * tf.to_float(samples[common.IMAGE]) - 1.0

    tf.logging.info('Performing single-scale test.')

    # import pb
    predictions = import_pb(FLAGS.pb, samples[common.IMAGE], INPUT_NODE_NAME, OUTPUT_NODE_NAME)

    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou_1.0'

    # Define the evaluation metric.
    metric_map = {}
    num_classes = dataset.num_of_classes
    metric_map['eval/%s_overall' % predictions_tag] = tf.metrics.mean_iou(
        labels=labels, predictions=predictions, num_classes=num_classes,
        weights=weights)
    # IoU for each class.
    one_hot_predictions = tf.one_hot(predictions, num_classes)
    one_hot_predictions = tf.reshape(one_hot_predictions, [-1, num_classes])
    one_hot_labels = tf.one_hot(labels, num_classes)
    one_hot_labels = tf.reshape(one_hot_labels, [-1, num_classes])
    for c in range(num_classes):
      predictions_tag_c = '%s_class_%d' % (predictions_tag, c)
      tp, tp_op = tf.metrics.true_positives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      fp, fp_op = tf.metrics.false_positives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      fn, fn_op = tf.metrics.false_negatives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)

      tp_fp_fn_op = tf.group(tp_op, fp_op, fn_op)
      iou = tf.where(tf.greater(tp + fn, 0.0),
                     tp / (tp + fn + fp),
                     tf.constant(np.NaN))
      metric_map['eval/%s' % predictions_tag_c] = (iou, tp_fp_fn_op)

    (metrics_to_values,
     metrics_to_updates) = contrib_metrics.aggregate_metric_map(metric_map)

    sess = tf.Session()

    sess.run(
        tf.group(
            tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
        )
    )

    while True:
        try:
            _ = sess.run(metrics_to_updates)
        except tf.errors.OutOfRangeError:
            break
    
    metrics_value = sess.run(metrics_to_values)
    for c in range(num_classes):
        predictions_tag_c = '%s_class_%d' % (predictions_tag, c)
        print(metrics_value['eval/%s' % predictions_tag_c])
    print(metrics_value['eval/%s_overall' % predictions_tag])


if __name__ == '__main__':
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
