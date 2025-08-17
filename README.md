# Brain Tumor Segmentation with U-Net + VGG16 (Kaggle - LGG Dataset)

This project implements a **U-Net architecture with a pre-trained VGG16 encoder** for brain tumor segmentation using the **LGG (Lower-Grade Glioma) dataset** from Kaggle.  
The model is designed to segment tumor regions in MRI images.

---

##  Project Steps

### 1. Data Loading and Preprocessing
- MRI images and their corresponding masks are loaded.
- Images are resized to **256×256 pixels**.
- Normalization is applied to scale pixel values between 0 and 1.
- Dataset is split into **training and validation sets**.

---

### 2. Model Architecture
- **U-Net** with **VGG16 encoder (pre-trained on ImageNet)**.
- Encoder (VGG16) extracts deep feature representations.
- Decoder reconstructs segmentation masks using upsampling and skip connections.
- Final layer: **Sigmoid activation** → binary segmentation (tumor vs. background).

---

### 3. Training
- Loss function: **Binary Cross-Entropy (BCE)** or **Dice Loss**.
- Optimizer: **Adam**.
- Metrics: **IoU (Intersection over Union)** and **Dice coefficient**.
- EarlyStopping and ModelCheckpoint are used for better performance.

---

### 4. Evaluation
- Model is tested on validation images.
- Performance is measured using **IoU** and **Dice coefficient**.
- Visual comparison between **predicted masks** and **ground truth masks**.

---

### 5. Results
- The model successfully segments brain tumors with high accuracy.
- **Predicted masks** closely match the ground truth.

---

##  Future Improvements
- Experiment with **ResNet, EfficientNet encoders**.
- Apply **data augmentation** (rotation, flipping, elastic transformation).
- Fine-tune hyperparameters for improved accuracy.
- Test on other brain tumor datasets.

---

##  Dataset
Dataset: [Kaggle LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

---

##  Requirements
```bash
tensorflow>=2.9
keras
numpy
matplotlib
scikit-learn
opencv-python
