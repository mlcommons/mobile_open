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

"""Extracts features for different models."""
import copy
import functools

import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

from nets.mobilenet import conv_blocks
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2

slim = contrib_slim

# Default end point for MobileNetv2 (one-based indexing).
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'
# Default end point for MobileNetv3.
_MOBILENET_V3_LARGE_FINAL_ENDPOINT = 'layer_17'
_MOBILENET_V3_SMALL_FINAL_ENDPOINT = 'layer_13'
# Default end point for EdgeTPU Mobilenet.
_MOBILENET_EDGETPU = 'layer_24'


def _mobilenet_v2(net,
                  depth_multiplier,
                  output_stride,
                  conv_defs=None,
                  divisible_by=None,
                  reuse=None,
                  scope=None,
                  final_endpoint=None):
  """Auxiliary function to add support for 'reuse' to mobilenet_v2.

  Args:
    net: Input tensor of shape [batch_size, height, width, channels].
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    conv_defs: MobileNet con def.
    divisible_by: None (use default setting) or an integer that ensures all
      layers # channels will be divisible by this number. Used in MobileNet.
    reuse: Reuse model variables.
    scope: Optional variable scope.
    final_endpoint: The endpoint to construct the network up to.

  Returns:
    Features extracted by MobileNetv2.
  """
  if divisible_by is None:
    divisible_by = 8 if depth_multiplier == 1.0 else 1
  if conv_defs is None:
    conv_defs = mobilenet_v2.V2_DEF
  with tf.variable_scope(
      scope, 'MobilenetV2', [net], reuse=reuse) as scope:
    return mobilenet_v2.mobilenet_base(
        net,
        conv_defs=conv_defs,
        depth_multiplier=depth_multiplier,
        min_depth=8 if depth_multiplier == 1.0 else 1,
        divisible_by=divisible_by,
        final_endpoint=final_endpoint or _MOBILENET_V2_FINAL_ENDPOINT,
        output_stride=output_stride,
        scope=scope)


def _mobilenet_v3(net,
                  depth_multiplier,
                  output_stride,
                  conv_defs=None,
                  divisible_by=None,
                  reuse=None,
                  scope=None,
                  final_endpoint=None):
  """Auxiliary function to build mobilenet v3.

  Args:
    net: Input tensor of shape [batch_size, height, width, channels].
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    divisible_by: None (use default setting) or an integer that ensures all
      layers # channels will be divisible by this number. Used in MobileNet.
    reuse: Reuse model variables.
    scope: Optional variable scope.
    final_endpoint: The endpoint to construct the network up to.

  Returns:
    net: The output tensor.
    end_points: A set of activations for external use.

  Raises:
    ValueError: If conv_defs or final_endpoint is not specified.
  """
  del divisible_by
  with tf.variable_scope(
      scope, 'MobilenetV3', [net], reuse=reuse) as scope:
    if conv_defs is None:
      raise ValueError('conv_defs must be specified for mobilenet v3.')
    if final_endpoint is None:
      raise ValueError('Final endpoint must be specified for mobilenet v3.')
    net, end_points = mobilenet_v3.mobilenet_base(
        net,
        depth_multiplier=depth_multiplier,
        conv_defs=conv_defs,
        output_stride=output_stride,
        final_endpoint=final_endpoint,
        scope=scope)

    return net, end_points


def mobilenet_v3_large_seg(net,
                           depth_multiplier,
                           output_stride,
                           divisible_by=None,
                           reuse=None,
                           scope=None,
                           final_endpoint=None):
  """Final mobilenet v3 large model for segmentation task."""
  del divisible_by
  del final_endpoint
  conv_defs = copy.deepcopy(mobilenet_v3.V3_LARGE)

  # Reduce the filters by a factor of 2 in the last block.
  for layer, expansion in [(13, 336), (14, 480), (15, 480), (16, None)]:
    conv_defs['spec'][layer].params['num_outputs'] /= 2
    # Update expansion size
    if expansion is not None:
      factor = expansion / conv_defs['spec'][layer - 1].params['num_outputs']
      conv_defs['spec'][layer].params[
          'expansion_size'] = mobilenet_v3.expand_input(factor)

  return _mobilenet_v3(
      net,
      depth_multiplier=depth_multiplier,
      output_stride=output_stride,
      divisible_by=8,
      conv_defs=conv_defs,
      reuse=reuse,
      scope=scope,
      final_endpoint=_MOBILENET_V3_LARGE_FINAL_ENDPOINT)


def mobilenet_edgetpu(net,
                      depth_multiplier,
                      output_stride,
                      divisible_by=None,
                      reuse=None,
                      scope=None,
                      final_endpoint=None):
  """EdgeTPU version of mobilenet model for segmentation task."""
  del divisible_by
  del final_endpoint
  conv_defs = copy.deepcopy(mobilenet_v3.V3_EDGETPU)

  return _mobilenet_v3(
      net,
      depth_multiplier=depth_multiplier,
      output_stride=output_stride,
      divisible_by=8,
      conv_defs=conv_defs,
      reuse=reuse,
      scope=scope,  # the scope is 'MobilenetEdgeTPU'
      final_endpoint=_MOBILENET_EDGETPU)


def mobilenet_v3_small_seg(net,
                           depth_multiplier,
                           output_stride,
                           divisible_by=None,
                           reuse=None,
                           scope=None,
                           final_endpoint=None):
  """Final mobilenet v3 small model for segmentation task."""
  del divisible_by
  del final_endpoint
  conv_defs = copy.deepcopy(mobilenet_v3.V3_SMALL)

  # Reduce the filters by a factor of 2 in the last block.
  for layer, expansion in [(9, 144), (10, 288), (11, 288), (12, None)]:
    conv_defs['spec'][layer].params['num_outputs'] /= 2
    # Update expansion size
    if expansion is not None:
      factor = expansion / conv_defs['spec'][layer - 1].params['num_outputs']
      conv_defs['spec'][layer].params[
          'expansion_size'] = mobilenet_v3.expand_input(factor)

  return _mobilenet_v3(
      net,
      depth_multiplier=depth_multiplier,
      output_stride=output_stride,
      divisible_by=8,
      conv_defs=conv_defs,
      reuse=reuse,
      scope=scope,
      final_endpoint=_MOBILENET_V3_SMALL_FINAL_ENDPOINT)


# A map from network name to network function.
networks_map = _mobilenet_v2


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           activation=tf.nn.relu6,
                           bn_decay=0.997,
                           bn_epsilon=None,
                           bn_renorm=None):
  """Defines the default MobilenetV2 arg scope.

  Args:
    is_training: Whether or not we're training the model. If this is set to None
      is_training parameter in batch_norm is not set. Please note that this also
      sets the is_training parameter in dropout to None.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: Standard deviation for initialization, if negative uses xavier.
    activation: If True, a modified activation is used (initialized ~ReLU6).
    bn_decay: decay for the batch norm moving averages.
    bn_epsilon: batch normalization epsilon.
    bn_renorm: whether to use batchnorm renormalization

  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': bn_decay,
  }
  if bn_epsilon is not None:
    batch_norm_params['epsilon'] = bn_epsilon
  if is_training is not None:
    batch_norm_params['is_training'] = is_training
  if bn_renorm is not None:
    batch_norm_params['renorm'] = bn_renorm
  dropout_params = {}
  if is_training is not None:
    dropout_params['is_training'] = is_training

  instance_norm_params = {
      'center': True,
      'scale': True,
      'epsilon': 0.001,
  }

  if stddev < 0:
    weight_intitializer = slim.initializers.xavier_initializer()
  else:
    weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
      weights_initializer=weight_intitializer,
      activation_fn=activation,
      normalizer_fn=slim.batch_norm), \
      slim.arg_scope(
          [conv_blocks.expanded_conv], normalizer_fn=slim.batch_norm), \
      slim.arg_scope([mobilenet.apply_activation], activation_fn=activation),\
      slim.arg_scope([slim.batch_norm], **batch_norm_params), \
      slim.arg_scope([mobilenet.mobilenet_base, mobilenet.mobilenet],
                     is_training=is_training),\
      slim.arg_scope([slim.dropout], **dropout_params), \
      slim.arg_scope([slim.instance_norm], **instance_norm_params), \
      slim.arg_scope([slim.conv2d], \
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
      slim.arg_scope([slim.separable_conv2d], weights_regularizer=None), \
      slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME') as s:
    return s


# A map from network name to network arg scope.
arg_scopes_map = mobilenet_v2.training_scope

# Names for end point features.
DECODER_END_POINTS = 'decoder_end_points'

# A dictionary from network name to a map of end point features.
networks_to_feature_maps = {
    DECODER_END_POINTS: {
        4: ['layer_4/depthwise_output'],
        8: ['layer_7/depthwise_output'],
        16: ['layer_14/depthwise_output'],
    },
}


# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
name_scope = 'MobilenetV2'

# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  num_channels = tf.shape(inputs)[-1]
  # We set mean pixel as 0 for the non-RGB channels.
  mean_rgb_extended = tf.concat(
      [mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
  return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
  """Map image values from [0, 255] to [-1, 1]."""
  preprocessed_inputs = (2.0 / 255.0) * tf.to_float(inputs) - 1.0
  return tf.cast(preprocessed_inputs, dtype=dtype)


_PREPROCESS_FN = _preprocess_zero_mean_unit_range

def mean_pixel():
  """Gets mean pixel value.

  _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
  The return values are used in a way that the padded regions after
  pre-processing will contain value 0.

  Returns:
    Mean pixel value.
  """
  return [127.5, 127.5, 127.5]


def extract_features(images,
                     output_stride=8,
                     depth_multiplier=1.0,
                     divisible_by=None,
                     final_endpoint=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     preprocessed_images_dtype=tf.float32,
                     num_classes=None,
                     global_pool=False,
                     nas_architecture_options=None,
                     nas_training_hyper_parameters=None,
                     use_bounded_activation=False):
  """Extracts features.

  Args:
    images: A tensor of size [batch, height, width, channels].
    output_stride: The ratio of input to output spatial resolution.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    divisible_by: None (use default setting) or an integer that ensures all
      layers # channels will be divisible by this number. Used in MobileNet.
    final_endpoint: The MobileNet endpoint to construct the network up to.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    preprocess_images: Performs preprocessing on images or not. Defaults to
      True. Set to False if preprocessing will be done by other functions. We
      supprot two types of preprocessing: (1) Mean pixel substraction and (2)
      Pixel values normalization to be [-1, 1].
    preprocessed_images_dtype: The type after the preprocessing function.
    num_classes: Number of classes for image classification task. Defaults
      to None for dense prediction tasks.
    global_pool: Global pooling for image classification task. Defaults to
      False, since dense prediction tasks do not use this.
    nas_architecture_options: A dictionary storing NAS architecture options.
      It is either None or its kerys are:
      - `nas_stem_output_num_conv_filters`: Number of filters of the NAS stem
        output tensor.
      - `nas_use_classification_head`: Boolean, use image classification head.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. It is either None or its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference. Currently,
      bounded activation is only used in xception model.

  Returns:
    features: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined
      by the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Unrecognized model variant.
  """
  arg_scope = arg_scopes_map(
      is_training=(is_training and fine_tune_batch_norm),
      weight_decay=weight_decay)
  features, end_points = get_network(
      preprocess_images, preprocessed_images_dtype, arg_scope)(
          inputs=images,
          depth_multiplier=depth_multiplier,
          divisible_by=divisible_by,
          output_stride=output_stride,
          reuse=reuse,
          scope=name_scope,
          final_endpoint=final_endpoint)

  return features, end_points


def get_network(preprocess_images,
                preprocessed_images_dtype=tf.float32, arg_scope=None):
  """Gets the network.

  Args:
    preprocess_images: Preprocesses the images or not.
    preprocessed_images_dtype: The type after the preprocessing function.
    arg_scope: Optional, arg_scope to build the network. If not provided the
      default arg_scope of the network would be used.

  Returns:
    A network function that is used to extract features.

  Raises:
    ValueError: network is not supported.
  """
  arg_scope = arg_scope or arg_scopes_map()
  def _identity_function(inputs, dtype=preprocessed_images_dtype):
    return tf.cast(inputs, dtype=dtype)
  def _none_function(inputs, dtype=preprocessed_images_dtype):
    return inputs
  if preprocess_images:
    preprocess_function = _PREPROCESS_FN
  elif preprocess_images is None:
    preprocess_function = _none_function
  else:
    preprocess_function = _identity_function
  func = networks_map
  @functools.wraps(func)
  def network_fn(inputs, *args, **kwargs):
    with slim.arg_scope(arg_scope):
      return func(preprocess_function(inputs, preprocessed_images_dtype),
                  *args, **kwargs)
  return network_fn
