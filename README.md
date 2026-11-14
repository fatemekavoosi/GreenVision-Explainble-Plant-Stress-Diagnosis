# 🌿 GreenVision: Lightweight MobileNetV2 & Explainable Grad-CAM Pipeline for Real-World Plant Disease Diagnosis

<img width="1778" height="1328" alt="gradcam_results" src="https://github.com/user-attachments/assets/be0752d6-b6cd-43a8-8714-7c7e758fc0de" />




## 📌 Project Overview

Automated visual monitoring in smart greenhouses and precision agriculture requires deep learning models that are both computationally lightweight (suitable for IoT Edge cameras) and transparent/interpretable for agronomists.

This repository implements an end-to-end Computer Vision and Explainable AI (XAI) pipeline in PyTorch trained on **real-world field leaf imagery (AI-Lab-Makerere/beans dataset). It leverages transfer learning with a lightweight MobileNetV2 backbone and Grad-CAM (Gradient-weighted Class Activation Mapping)** to classify foliar diseases and visually localize symptomatic lesions.



## 🚀 Key Features & Architecture

1. **Real-World Agricultural Dataset (Beans Leaf Dataset):**
  Evaluates balanced field subsets (450 training and 150 validation/test images) across 3 real-world classes: Healthy, Angular Leaf Spot, and Bean Rust.
  Includes data augmentation (random resized cropping, horizontal flips, rotation, and color jittering) to handle natural lighting variations.

2. **Lightweight Edge-AI Backbone (MobileNetV2):**
  Fine-tunes an ImageNet-pretrained MobileNetV2 (~2.2M parameters) utilizing depthwise separable convolutions for low-latency inference on resource-constrained edge nodes.

3. Explainable AI (Grad-CAM Lesion Localization):
  Computes spatial activation heatmaps from the final convolutional block (model.features[-1]), demonstrating that classification decisions directly focus on necrotic spots and rust pustules.

4. Edge Hardware Profiling:
  Benchmarks macro F1-score, accuracy, model size (MB), parameter count, and per-image inference latency across both CPU and GPU environments.



## 🛠️ Tech Stack

 Deep Learning & Vision: PyTorch, Torchvision (MobileNetV2)
 Dataset & Preprocessing: HuggingFace datasets, Pillow, NumPy
 Evaluation & Visualization: Scikit-Learn, Matplotlib



## 📂 Repository Structure

 plant_vision_XAI_pipeline.ipynb : Complete modular notebook containing real dataset loading, MobileNetV2 fine-tuning, Grad-CAM visualization, and CPU/GPU latency profiling. gradcam_results.png : 4-column visual dashboard  showing real test leaves, preprocessed inputs, Grad-CAM heatmaps, and confidence overlays.
 requirements.txt : Required Python dependencies.

---

## ⚙️ How to Run

1. Clone the repository:
   git clone https://github.com/fatemekavoosi/GreenVision-Explainble-Plant-Stress-Diagnosis.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run Notebook:
   Open and execute plant_vision_XAI_pipeline.ipynb in Jupyter Notebook or Google Colab.
