# 🧠 Ensemble-Based Brain Tumor Classification Using Pretrained CNN Architectures

## 📌 Abstract

Accurate classification of brain tumors from medical imaging is a critical task in computer-aided diagnosis. This repository presents a deep learning-based approach for multi-class brain tumor classification using an ensemble of pretrained convolutional neural networks. Specifically, EfficientNetV2B3, DenseNet121, EfficientNetB4, and Xception architectures are employed as feature extractors with a custom classification head. Predictions from individual models are combined using a simple averaging ensemble strategy. The implementation follows a deterministic and reproducible pipeline derived directly from experimental notebook-based research.

---

## 🚀 Key Contributions

* Implementation of **four pretrained CNN architectures** for medical image classification
* A unified **ensemble framework using prediction averaging**
* Modular and reproducible pipeline derived from experimental notebook
* Clean separation of:

  * Data processing
  * Model definition
  * Training
  * Evaluation
* Lightweight and extensible design for future research

---

## 🧪 Method Overview

### Dataset Handling

* Images are loaded using **OpenCV (`cv2`)**
* Data is collected from:

  * `Training/`
  * `Testing/`
* Both are merged into a single dataset and then split using:

  * `train_test_split (10%)`

---

### Preprocessing

* Image resizing: **150 × 150**
* No normalization applied (pixel range remains `[0–255]`)
* Labels encoded using:

  * Index mapping
  * One-hot encoding

---

### Model Architecture

Each model follows the same structure:

* Pretrained backbone (`include_top=False`)
* Frozen feature extractor
* Classification head:

  * GlobalAveragePooling2D
  * Dense (128, ReLU)
  * Dropout (0.5)
  * Dense (Softmax, 4 classes)

#### Models Used:

* EfficientNetV2B3
* DenseNet121
* EfficientNetB4
* Xception

---

### Training Strategy

* Optimizer: `Adam`
* Loss: `categorical_crossentropy`
* Metric: `accuracy`
* Epochs: 20
* Batch size: 32
* Validation:

  * Uses test split (`X_test`, `Y_test`)

Callbacks:

* EarlyStopping
* ReduceLROnPlateau

---

### Ensemble Method

* Predictions from all models are averaged using **Simple Aceraging**

* Final prediction via `argmax`

---

## 📁 Repository Structure

```bash
brain-tumor-ensemble/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── Ensemble_Brain_Tumor_Cleaned.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── effnetv2b3.py
│   │   ├── densenet121.py
│   │   ├── effnetb4.py
│   │   ├── xception.py
│   │   └── ensemble.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── logger.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── experiments/
├── tests/
└── docs/
```

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd brain-tumor-ensemble

pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Training

```bash
python scripts/train.py \
    --data_dir path/to/dataset \
    --model effnetv2b3
```

---

### 🔹 Evaluation

```bash
python scripts/evaluate.py \
    --data_dir path/to/dataset
```

---

### 🔹 Inference

```bash
python scripts/predict.py \
    --image path/to/image.jpg
```

---

## 📊 Results

The results are obtained directly from the notebook implementation. The following metrics are computed:

* Accuracy
* Confusion Matrix
* Classification Report

### 🔹 Model-wise Performance

| Model             | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|------------------|----------|----------|--------|----------|
| EfficientNetV2B3 | 95.11    | 95    | 95  | 95    |
| EfficientNetB3   | 80.43    | 82    | 82  | 80    |
| Xception         | 94.50    | 94    | 95  | 95    |
| **Ensemble (Avg)** | **96.02** | **96** | **96** | **96** |

### 🔹 Hyperparameter Selection

The hyperparameters used in this study are directly derived from the original experimental notebook and kept consistent across all models to ensure fair comparison.

#### 📌 Training Configuration

| Parameter        | Value |
|-----------------|------|
| Optimizer       | Adam (default settings) |
| Loss Function   | Categorical Crossentropy |
| Batch Size      | 32 |
| Epochs          | 20 |
| Validation Split| 10% (via train_test_split) |
| Input Size      | 150 × 150 × 3 |

---

#### 📌 Model Architecture Configuration

| Component                  | Setting |
|---------------------------|--------|
| Pretrained Weights        | ImageNet |
| Include Top Layer         | No (`include_top=False`) |
| Feature Extractor         | Frozen (no fine-tuning) |
| Pooling Layer             | GlobalAveragePooling2D |
| Dense Layer               | 128 units (ReLU) |
| Dropout                   | 0.5 |
| Output Layer              | Softmax (4 classes) |

---

#### 📌 Callbacks

| Callback              | Configuration |
|----------------------|--------------|
| EarlyStopping        | monitor=val_loss, patience=10 |
| ReduceLROnPlateau    | factor=0.3, patience=5, min_lr=1e-6 |

---

#### 📌 Ensemble Strategy

| Parameter        | Value |
|-----------------|------|
| Ensemble Type   | Simple Averaging |
| Weighting       | Equal weights |
| Final Decision  | Argmax over averaged predictions |

---

## 🔁 Reproducibility

This repository ensures reproducibility through:

* Config-driven setup (`configs/config.yaml`)
* Seed control (`src/utils/seed.py`)
* Deterministic pipeline (optional)

To reproduce results:

1. Use the same dataset structure
2. Keep config values unchanged
3. Run training script

---

## 📌 Future Work

* Proper dataset split (Train / Val / Test)
* Model fine-tuning (unfreezing layers)
* Weighted or stacking-based ensemble
* Integration with TensorBoard / W&B
* Data normalization and augmentation
* Clinical validation

---
## 🙏 Acknowledgements

* TensorFlow / Keras
* Pretrained model contributors
* Open-source medical imaging datasets
