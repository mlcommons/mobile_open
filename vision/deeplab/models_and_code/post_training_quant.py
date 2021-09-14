"""Post-training quantization on DeeplabV3-seg using official TFLiteConverter."""

import pathlib

import cv2
import numpy as np
import six
import tensorflow as tf
import tqdm


flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')
flags.DEFINE_string('graph_def', None, 'Name of the input graph def file.')
flags.DEFINE_string('out_tflite', None, 'Name of the output tflite.')
flags.DEFINE_enum('input_type', None, ['float', 'uint8'], 'Which type of input is used.')
flags.DEFINE_integer('sample_num', 500,
                     'The number of samples used in quantization.')


def image_generator():
  # Generates an iterator over images
  for idx in tqdm.tqdm(range(FLAGS.sample_num)):
    input_path = str(pathlib.Path(FLAGS.dataset_dir) / f'img_{idx}.png')
    input_data = np.expand_dims(
      np.array(
        cv2.cvtColor(cv2.imread(input_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
      ), 0
    )
    yield[(input_data / 127.5 - 1.0).astype(np.float32)]


def main(unused_argv):

  input_arrays = ["MobilenetV2/MobilenetV2/input"]
  output_arrays = ["ArgMax"]


  converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            FLAGS.graph_def, input_arrays, output_arrays)

  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = image_generator

  if FLAGS.input_type == 'float':
    converter.inference_input_type = tf.float32
  elif FLAGS.input_type == 'uint8':
    converter.inference_input_type = tf.uint8
  else:
    raise ValueError(f'Input type {FLAGS.input_type} not supported. Should be either float or uint8.')

  converter.inference_output_type = tf.uint8
  tflite_quant_model = converter.convert()
  open(FLAGS.out_tflite, "wb").write(tflite_quant_model)


if __name__ == '__main__':
  flags.mark_flag_as_required('dataset_dir')
  tf.compat.v1.app.run()
