# DIP Assignment Questions and Answers

This repository contains all DIP assignment questions and their solutions.

---

### Questions

1. Implement and compare methods for grayscale quantization (uniform, clustering/median-cut, octree) on images converted to grayscale; evaluate visual quality and trade-offs at a fixed number of levels (e.g., 16).

2. Implement K-means clustering for image color quantization (rate–distortion): cluster pixels in color space, reconstruct the image from centroids for a chosen k (e.g., 5), and compare visual quality and distortion trade-offs.

3. Implement grayscale conversion followed by frequency-domain sampling using FFT masks (1/2, 1/4, 1/8, 1/16 of low frequencies retained) and spatial resolution sampling (image downscaled by the same ratios), and compare their visual effects.

4. HDR: Compute HDR via recovery of irradiance map and a global tone mapping algorithm. Sample photos in GitHub. Or, use this which implements 1. Estimation of the Camera Response Function, 2. Computation of the irradiance map & 3. Tone Mapping.

5. Spatial Filtering: Implement 5x5 and 20x20 box filters with and without normalization for Torgya - Arunachal Festival.jpg. Note, this is a color image, not a greyscale. Compute sigma, use it to compute filter size and then, apply a separable Gaussian filter and a separable normalized Gaussian filter.

6. Bit-plane Splicing: Take your photo in low light and in bright light. Compute bit-planes for each. Reconstruct the original image using the union of three lowest bitplanes. Difference the union from the original image.
