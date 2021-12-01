# MOSAIC

## Mobile Semantic Image Segmentation Model for MLPerf v2.0

MOSAIC is a next-generation neural network architecture for efficient and accurate semantic image segmentation on mobile devices.
MOSAIC is designed using commonly supported neural operations by diverse mobile hardware platforms for flexible deployment across various mobile platforms.
With a simple asymmetric encoder-decoder structure which consists of an efficient multi-scale context encoder
and a light-weight hybrid decoder to recover spatial details from aggregated information, MOSAIC achieves new state-of-the-art performance while balancing
accuracy and computational cost.
Deployed on top of a tailored feature extraction backbone based on a searched classification network, MOSAIC achieves a 5\% absolute accuracy gain with similar
or lower latency compared to the current industry standard MLPerf v1.0 models and state-of-the-art architectures.

### Performance on ADE20K-Top31
|Model | mIOU | MAdds(B) | Latency(ms) on Pixel 4 | 
|------|------------------:|-----------------:|-----------------:|
|MLPerf v1.0 | 54.80% | 2.7 | 219 (CPU) / 134 (GPU) / 109 (DSP) / 165.6 (EdgeTPU) |
|MNV3-Small + LR-ASPP | 52.57% | 0.6 | 99.7 (CPU) / 67.1 (GPU) / 71.3 (DSP) / 66.2 (EdgeTPU) |
|MNV3-Large + LR-ASPP | 55.36% | 1.43 | 176.1 (CPU) / 63.7 (GPU) / 135 (DSP) / 118.8 (EdgeTPU) |
|MOSAIC | 60.10% | 2.98 | 116 (CPU) / 107 (GPU) / 39 (DSP) / 56.6 (EdgeTPU) |
