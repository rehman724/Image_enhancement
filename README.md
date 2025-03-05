# Image_enhancement
## Project Overview

This project focuses on image enhancement and processing using machine learning techniques. The primary goal is to enhance blurry images by applying deep learning-based deblurring methods.

## Features
Load and preprocess blurry images.
Apply a deep learning model for image deblurring.
Visualize and compare original vs. enhanced images.
Save the processed images for further analysis.

## Requirements
To run this project, install the following dependencies:
pip install numpy pandas matplotlib opencv-python tensorflow keras torch torchvision

## Implementation Steps
Load Dataset: Load blurry images and their corresponding sharp images.
Preprocess Images: Resize images, normalize pixel values, and convert to tensors.
Train a Model: Use a CNN-based model (e.g., GANs, UNet, SRCNN) for deblurring.
Evaluate the Model: Compare results using PSNR and SSIM metrics.
Visualize Results: Display before-and-after images using Matplotlib.

## Usage
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("sample_blurry.jpg")
processed_image = deblur_model.predict(image)

plt.imshow(processed_image)
plt.show()

## Issues & Troubleshooting
If you encounter the error:
Invalid shape (1, 128, 128, 3) for image data
plt.imshow(image[0])  # Remove batch dimension
