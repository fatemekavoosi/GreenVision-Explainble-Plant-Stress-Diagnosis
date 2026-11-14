# 🌿 GreenVision: Lightweight CNN & Explainable Grad-CAM Pipeline for Smart Greenhouse Plant Stress Diagnosis

<img width="1690" height="1276" alt="gradcam-results" src="https://github.com/user-attachments/assets/bd6d1bef-f2de-4f90-b17f-03a23333292a" />


## 📌 Project Overview
Automated visual monitoring in smart greenhouses and precision agriculture requires deep learning models that are both computationally lightweight (suitable for IoT Edge cameras) and transparent/interpretable for agronomists. 

This repository implements an end-to-end Computer Vision and Explainable AI (XAI) pipeline in PyTorch. It combines classical image preprocessing with a resource-efficient Depthwise Separable CNN and Grad-CAM (Gradient-weighted Class Activation Mapping) to classify plant foliar diseases and visually localize symptomatic regions without requiring bounding-box annotations.



## 🚀 Key Features & Architecture
1. Classical Image Processing Preprocessing:
    Denoising: Gaussian/Median spatial filtering to suppress sensor and environmental noise.
    Contrast Enhancement: Adaptive histogram equalization (CLAHE) to normalize greenhouse lighting variations.
    ROI Segmentation: HSV color-space thresholding and morphological operations to isolate leaf tissue from complex backgrounds.
2. Lightweight Edge-CNN Architecture:
    Utilizes Depthwise Separable Convolutions, Batch Normalization, and Global Average Pooling to drastically reduce parameter count and FLOPs for real-time edge inference.
3. Explainable AI (Grad-CAM Localization):
    Extracts gradients from the final convolutional layer to generate class-discriminative heatmaps, proving that the network focuses on actual necrotic spots and fungal lesions rather than background artifacts.
4. Edge Hardware Profiling:
    Evaluates model memory footprint (MB), parameter count, and per-image inference latency (ms).



## 🛠️ Tech Stack
 Deep Learning: PyTorch
 Image Processing: OpenCV, NumPy, SciPy
 Evaluation & Visualization: Scikit-Learn, Matplotlib, Seaborn



## 📂 Repository Structure
 plant_vision_XAI_pipeline.ipynb: Complete modular notebook containing data synthesis, classical image processing, lightweight CNN training, Grad-CAM computation, and edge profiling.
 gradcam_results.png: Visual dashboard comparing raw inputs, preprocessed masks, Grad-CAM heatmaps, and confidence overlays.
 requirements.txt: Required Python dependencies.



## ⚙️ How to Run
1. Clone the repository:
   git clone https://github.com/fatemekavoosi/GreenVision-Explainable-Plant-Stress-Diagnosis.git
2. Install dependencies:
   pip install -r requirements.txt
3. Open and execute plant_vision_XAI_pipeline.ipynb in Jupyter Notebook or Google Colab.
