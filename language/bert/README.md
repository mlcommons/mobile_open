# MobileBERT with SQUAD 1.1

There are TensorFlow SavedModel representations in the [`models_and_code` directory](models_and_code/checkpoints),
each with two corresponding .tflite models. The SavedModel instances were trained against the SQUAD 1.1 dataset, with
sequence length 384, using the [Mobile BERT TensorFlow source](https://github.com/google-research/google-research/commit/ec663b464b060d24d9f86c6ac155f6734d82f0e1).

## [Float](models_and_code/checkpoints/float)
Contains a floating point, [pre-trained SavedModel](models_and_code/checkpoints/float/saved_model.pb)
with two corresponding .tflite models.

The TensorFlow Lite models are:
 * [float.tflite](models_and_code/checkpoints/float/mobilebert_float_384.tflite) - Floating point .tflite model.
 * [float_gpu.tflite](models_and_code/checkpoints/float/mobilebert_float_384_gpu.tflite) - Floating point .tflite model with several mathematically equivalent op replacements for TensorFlow Lite GPU compatibility.

## [Quantized](models_and_code/checkpoints/quant)
Contains a partially quantized, [pre-trained SavedModel]((models_and_code/checkpoints/quant/saved_model.pb)
with two .tflite models. This TF model was trained using *partial* quantization-aware training, and
the corresopnding .tflite models were finalized using post-training quantization with the first 100
questions from the SQUAD 1.1 dataset.

Note that this partially quantized TensorFlow model also differs from the corresponding floating
point model in several minor ways:
 * `min` & `max` used for clipping
 * `requant` nodes inserted for input/input
 * `relu6` used instead of `relu`

The TensorFlow Lite models are:
 * [quant.tflite](models_and_code/checkpoints/float/mobilebert_int8_384.tflite) - Quantized (int8, per-channel) .tflite model.
 * [quant_nnapi.tflite](models_and_code/checkpoints/float/mobilebert_int8_384_nnapi.tflite) - Quantized (int8, per-channel) .tflite model with several mathematically equivalent op replacements for NNAPI compatibility.

The ONNX model:
[ONNX model](models_and_code/checkpoints/quant/mobilebert_squad11_int8_qdq_89.4f1.onnx) is based on original [model weights](https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT.tar.gz) . Imported this model to PyTorch using Huggingface and did QAT using NNCF and exported INT8 weights to ONNX format.
## Accuracy (F1 Score)

|Model | Tensorflow(Lite) | ONNX|
|------|------------------|----|
|FP32              |  90  ||
|INT8 (QAT + PTQ)  |  88  |[89.4](models_and_code/checkpoints/quant/mobilebert_squad11_int8_qdq_89.4f1.onnx)|
|||

