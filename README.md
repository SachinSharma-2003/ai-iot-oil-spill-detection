# 🌊 AI-Enabled Marine Oil Surveillance System  
### Automated SAR Neural Analysis & Operational Response Protocol

---

## 📌 Project Overview

Oil spills pose a critical threat to marine ecosystems, coastal livelihoods, and national economies, demanding rapid detection and precise response planning.  
This project presents an **industrial-grade AI surveillance platform** that automates the **detection and segmentation of marine oil spills** using **Synthetic Aperture Radar (SAR) satellite imagery**.

Leveraging a **deep learning U-Net architecture**, the system transforms raw radar backscatter into **actionable maritime intelligence**, delivering:
- Pixel-level oil spill segmentation  
- Accurate areal estimation in square kilometers ($km^2$)  
- Severity-based operational response guidance for maritime authorities  

The platform is deployed as a **real-time interactive dashboard** optimized for command-center-level monitoring.

---

## 🎯 Objectives

- **Automated Target Acquisition**  
  Eliminate manual interpretation errors by identifying oil slicks through neural signal decoding.

- **Precision Segmentation**  
  Perform pixel-level mask extraction to accurately distinguish hydrocarbons from open-water backscatter.

- **Areal Quantification**  
  Compute the physical extent of oil spills in square kilometers ($km^2$) to support effective resource allocation.

- **Operational Intelligence**  
  Provide severity-based risk classification (Low / Medium / High) with response recommendations for marine departments.

---

## 🗂 Dataset

- **Input**: Grayscale SAR satellite images  
- **Output**: Binary segmentation masks (Oil Spill / Non-Oil)  
- **Training Samples**: 6,455 images  
- **Validation Samples**: 1,615 images  

The dataset captures diverse spill geometries, noise conditions, and environmental backscatter variations.

---

## ⚙️ Methodology

### 1️⃣ Preprocessing
- **Speckle Reduction**: Median filtering to suppress SAR-specific grain noise  
- **Standardization**: Resize to $256 \times 256$ and normalize pixel values to $[0,1]$  
- **Binarization**: Thresholded mask preparation for supervised learning  

---

### 2️⃣ Model Architecture
- **U-Net Encoder–Decoder**  
  Symmetrical CNN architecture for simultaneous contextual understanding and spatial localization.
- **Skip Connections**  
  Preserve high-resolution spatial features across network depth.
- **Sigmoid Activation**  
  Produces pixel-wise probability maps for oil spill segmentation.

---

### 3️⃣ Training & Evaluation
- **Optimizer**: Adam  
- **Training Strategy**: Mixed-precision training for accelerated convergence  
- **Loss Function**: Binary Cross-Entropy  
- **Evaluation Metrics**:
  - Dice Coefficient
  - Intersection over Union (IoU)

---

## ⚓ Professional Dashboard Features

- **Glassmorphism UI**  
  Premium dark-mode interface designed for maritime surveillance environments.

- **Areal Scaling**  
  Real-time conversion of segmentation output into physical spill area ($km^2$).

- **Neural Latency Monitoring**  
  Live AI inference timing (typically < 1.0s per image).

- **Visual Legends**  
  - Binary Mask: **White = Oil Detected**  
  - Heatmap: **Red = Oil Spill | Blue = Water**

- **Dynamic Reset Handling**  
  Automatic session cleanup on new SAR uplink to ensure UI stability.

---

## 📊 Results

- **Mean Dice Score**: 0.70  
- **Mean IoU**: 0.57  
- **AI Confidence**: Dynamic probability estimation per image  
- **Stability**: Robust generalization across noisy SAR conditions  

The model demonstrates reliable segmentation performance suitable for operational monitoring.

---

## 🚀 Future Enhancements

- **Attention U-Net**  
  Improved boundary detection for low-contrast oil slicks.

- **GIS Integration**  
  Real-time geospatial mapping and GPS-based vessel dispatch.

- **Temporal Drift Tracking**  
  Time-series analysis to forecast oil slick movement using ocean current and wind data.

---

## 🛠 Technologies Used

- **Programming Language**: Python  
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib  
- **Deployment**: Streamlit (Custom Premium CSS)  
- **Environment**: Google Colab, Hugging Face Hub  

---

## 👨‍🎓 Author

**Sachin Sharma A**  
Final Year B.Tech (Information Technology)  
Infosys Springboard Intern  
**Specialization**: AI–IoT Enabled Marine Detection Systems  

---

## 📄 License
This project is intended for academic, research, and demonstration purposes.
