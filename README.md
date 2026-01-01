# AI-Driven System for Oil Spill Identification and Monitoring

## 📌 Project Overview
Oil spills cause severe damage to marine ecosystems and coastal economies. Manual inspection of satellite imagery is time-consuming and error-prone. This project presents an AI-driven system that automatically detects and segments oil spill regions from Synthetic Aperture Radar (SAR) satellite images using a U-Net deep learning model.

The system helps authorities identify spill locations quickly and supports faster response and cleanup operations.

---

## 🎯 Objectives
- Automatically detect oil spill regions from SAR images
- Perform pixel-level segmentation of oil and non-oil areas
- Reduce manual effort and improve monitoring efficiency
- Provide accurate and interpretable segmentation outputs

---

## 🗂 Dataset
- **Input**: SAR satellite images (grayscale)
- **Output**: Binary masks (oil spill vs non-oil)
- **Training samples**: 6455 images
- **Validation samples**: 1615 images

---

## ⚙️ Methodology
1. **Dataset Preparation**
   - Extract images and masks
   - Remove noise files (`__MACOSX`, `.DS_Store`)

2. **Preprocessing**
   - Median filtering to reduce SAR speckle noise
   - Resize to 256×256
   - Normalize images to [0,1]
   - Binarize masks

3. **Model Architecture**
   - U-Net encoder–decoder architecture
   - Skip connections to preserve spatial details
   - Sigmoid activation for pixel-wise probability output

4. **Training Strategy**
   - `tf.data.Dataset` for disk-based data loading
   - Mixed precision training for faster computation
   - Binary Cross-Entropy loss
   - Adam optimizer

5. **Evaluation**
   - Quantitative: Dice Score, IoU
   - Qualitative: Visual comparison of predictions

---

## 📊 Results
- **Mean Dice Score**: 0.70  
- **Mean IoU**: 0.57  
- Training and validation losses show stable convergence
- Model generalizes well without overfitting

---

## 🖼 Sample Outputs
The model successfully segments oil spill regions and captures spatial boundaries effectively, even in noisy SAR images.

---

## 🚀 Future Enhancements
- Attention U-Net for improved boundary detection
- Data augmentation for robustness
- Post-processing (morphological operations)
- Real-time visualization using GIS / Google Maps

---

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Google Colab

---

## 👨‍🎓 Author
Final Year B.Tech – Information Technology  
Academic Project
