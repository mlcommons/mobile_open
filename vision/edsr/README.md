# EDSR for Mobile Super-Resolution

## Overview

This directory provides the TensorFlow 2.0 implementation of [EDSR](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf) (Lim et al., CVPR Workshops 2017).
While the original paper proposes EDSR with (64 channels, 16 blocks) and (256 channels, 32 blocks),
we provide EDSR (scale 2) with (64 channels, 5 blocks: **f64b5**) and (32 channels, 5 blocks: **f32b5**) for lightweight mobile execution. We also note that there are some minor changes in the model architecture for better optimization.

## Directory structure
```bash
models_and_checkpoints
 |- checkpoints
 |   |- f32b5
 |   |   |- ckpt
 |   |   |   `- # saved_model for the f32b5 model (FP32)
 |   |   |
 |   |   |- ckpt_qat
 |   |   |   `- # saved_model for the f32b5 model (UINT8, QAT)
 |   |   |
 |   |   |- model.txt
 |   |   |   `- # Detailed model architecture for the f32b5 model (FP32)
 |   |   |
 |   |   `- model_qat.txt
 |   |   |   `- # Detailed model architecture for the f32b5 model (UINT8, QAT)
 |   |
 |   `- f64b5
 |       `- # Same as the f32b5
 |
 `- tflite
     `- # Contains *.tflite files for f32b5/f64b5 models (FP32 and UINT8)
```

## Performance
