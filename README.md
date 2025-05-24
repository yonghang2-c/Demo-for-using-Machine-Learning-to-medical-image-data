# Demo-for-using-Machine-Learning-to-medical-image-data
This demo shows a brief methodology for extracting basic medical features from medical images and using ML to quantify its associations. The  NIH-Chest-X-ray-dataset was used and the relationship among Lesion Ratio from X-ray dataset and other medical features was accessed. 

## Chest X-ray Segmentation & Feature Extraction Demo

This repository contains a demo pipeline for:

1. **Chest X-ray image segmentation** using a U-Net architecture (via `segmentation-models-pytorch`).
2. **Spatial feature extraction** (lesion ratio, number of lesions, mean lesion area, mean centroid distance).
3. **Integration with clinical metadata** and **regression analysis** to predict lesion ratio from clinical and spatial features.
4. **Hyperparameter optimization** for multiple regression models (XGBoost, LightGBM, Random Forest) using Optuna.
5. **Model evaluation & visualization** (5-fold CV scatter plot, SHAP feature importance).

---

### 📋 Prerequisites

* **Python 3.10+**
* GPU (CUDA) or CPU

Install required Python packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** should include:

import os

import random

import numpy as np

import pandas as pd

from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

import shap

import torch

import torch.nn as nn

from torch.amp import GradScaler, autocast

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import segmentation_models_pytorch as smp

from segmentation_models_pytorch.losses import DiceLoss

from skimage.measure import label as sk_label, regionprops

from tqdm import tqdm

import shutil

import optuna

import json

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from joblib import dump

import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from math import sqrt

---

### 🗂 Repository Structure

```
├── README.md                 # This document
├── demo.py                   # demo code
├── requirements.txt          # Python package dependencies
├── NIH-Chest-X-ray-dataset   # Example public dataset from NIH: [https://nihcc.app.box.com/v/ChestXray-NIHCC]
└── outputs/                  # Generated artifacts (models, CSVs, plots)
```

---

### ⚙️ Medical images preparation (Chest X-ray) and Configuration

1. Down the medical imgaes from https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/tree/main/data/images
2. Place original X-ray folders (`images1`, `images2`, ..., `images12`) into the "NIH-Chest-X-ray-dataset" folder
3. Create a new folder in "NIH-Chest-X-ray-dataset" folder: images for the combination of downloaded images
4. In `demo.py`, update `DATA_DIR` to local path:
       DATA_DIR = "/path/to/your/demo"

---

### 🚀 Usage

Run the full pipeline:

```bash
python demo.py
```

This will:

1. **Copy** images into `images/`.
2. **Train** a U-Net model for segmentation (saving best model as `best_unet.pth`).
3. **Extract** spatial features to `spatial_features.csv`.
4. **Merge** with clinical metadata and run **Optuna** optimization for three ML models: XGBoost, LightGBM, and Random Forest.
5. **Save** best hyperparameters (`best_hyperparams.csv`), models (`*_best_model.joblib`), metrics, and visualizations:

   * `training_history.csv`
   * `cv_scatter_<model>.png`
   * `shap_importance_<model>.png`
   * `shap_values_best_<model>.csv`

---

### 📊 Results & Visualization

* **Training history**: `training_history.csv`
* **CV scatter plot**: `cv_scatter_<best_model>.png`
* **SHAP feature importance**: `shap_importance_<best_model>.png`
* **SHAP values**: `shap_values_best_<best_model>.csv`

Example:

Results/cv_scatter_rf.png

Results/shap_importance_rf.png

---

### 🛠️ Extending & Customization

* **Model architecture**: Switch encoder or architecture in the U-Net instantiation.
* **Transforms**: Modify `SegmentationTransform` for additional augmentations.
* **Hyperparameter search**: Adjust `n_trials` or parameter ranges in the Optuna `objective`.
* **Batch size, epochs, learning rate**: Edit constants in `demo.py`.

---

### 📜 License

This project is a demo and no restrictions on the use. 

### 📜 Acknowledgement

Acknowledge that the NIH Clinical Center for providing the public data: https://nihcc.app.box.com/v/ChestXray-NIHCC
More details can be referred to: 
 
 @inproceedings{Wang_2017,
    doi = {10.1109/cvpr.2017.369},
    url = {https://doi.org/10.1109%2Fcvpr.2017.369},
    year = 2017,
    month = {jul},
    publisher = {{IEEE}
},
    author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
    title = {{ChestX}-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
    booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}


### ✉️ Contact

For questions or suggestions, please open an issue or contact `lai.yonghang@nies.go.jp`.

