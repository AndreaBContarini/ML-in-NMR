# Machine Learning Applications in NMR Imaging for Osteoporosis Detection

This project uses machine learning techniques to analyze MRI images for detecting osteoporosis. It leverages deep learning models and dimensionality reduction techniques to classify patients as healthy or osteoporotic based on MRI data.

---

## Project Overview

- **Objective**: Develop a classifier to distinguish between healthy and osteoporotic patients using MRI data.
- **Dataset**:
  - MRI images of 33 patients: 12 healthy and 21 osteoporotic.
  - Images include \(T_1\)-weighted (10 repetition times) and \(T_2\)-weighted (9 echo times) acquisitions.
  - Data preprocessed into normalized `.npy` format for efficient computation.

- **Methods**:
  - Preprocessing includes normalization, dimensional reduction, and latent space extraction.
  - Convolutional neural networks (CNNs) for classification.
  - Jackknife cross-validation for robust model evaluation.
  - Dimensional reduction using PCA, ICA, t-SNE, and UMAP.

---

## Files in the Repository

### 1. **`Neural_Network.py`**
   - Main Python script implementing:
     - Preprocessing of MRI data.
     - Convolutional neural network for classification.
     - Training loop with Jackknife cross-validation.
     - Feature extraction and dimensionality reduction.

### 2. **`ML applications in NMR.pptx`**
   - Presentation providing:
     - MRI and NMR fundamentals.
     - Model architecture and training methodology.
     - Visualization of latent space and classification results.
     - Conclusions and future directions.

---

## Methods and Algorithms

### MRI Data Processing
- **Normalization**:
  - \(T_2\)-weighted images normalized against the first slice (\(T_E = 25 \,ms\)) to reduce absolute intensity differences.
  - Removes the first slice post-normalization for consistent input shape.

- **Dataset Structure**:
  - \(T_1\): \( (1, 512, 512, 10) \)
  - \(T_2\): \( (6-9, 512, 512, 9) \)

### Model Architecture
- **Convolutional Neural Network (CNN)**:
  - Separate convolutional pipelines for \(T_1\) and \(T_2\) images.
  - 3D convolutions for feature extraction across temporal dimensions.
  - Fully connected layers for classification.

- **Jackknife Cross-Validation**:
  - Exclude one patient per iteration for validation.
  - Train on remaining data.
  - Aggregate results for performance metrics.

### Dimensionality Reduction
- **Independent Component Analysis (ICA)**:
  - Extract statistically independent components.
  - Analyze latent space separability.
- **Principal Component Analysis (PCA)**:
  - Reduce data dimensionality while retaining variance.
- **t-SNE & UMAP**:
  - Visualize non-linear relationships in latent space.

---

## Results and Insights

- **Classification Performance**:
  - Accuracy: ~55%
  - Precision: ~62%
  - Recall: ~76%
  - F1-Score: ~68%

- **Latent Space Visualization**:
  - ICA, PCA, t-SNE, and UMAP provided limited class separability.
  - Suggests dataset limitations and model refinement needs.

- **Challenges**:
  - Dataset imbalance (21 osteoporotic vs. 12 healthy).
  - Small dataset size affects generalizability.

---

## How to Use

### 1. Requirements
- **Programming Language**: Python 3.8+
- **Libraries**:
  - `torch`, `numpy`, `nibabel`, `matplotlib`, `sklearn`, `umap-learn`

### 2. Setup
Install dependencies:
```bash
pip install torch numpy nibabel matplotlib scikit-learn umap-learn
