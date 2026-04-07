# MAGIC Gamma Telescope Data Modeling

This project builds and compares supervised learning models to classify high-energy particle events as either:

- `g` (gamma signal)
- `h` (hadron background)

The work is based on the **MAGIC Gamma Telescope** dataset and is implemented in the notebook `magic_gamma.ipynb`.

## Project Goal

Ground-based Cherenkov telescopes detect light patterns produced by particle showers in the atmosphere. The goal is to use image-derived features to separate true gamma-ray events (signal) from hadronic cosmic-ray events (background).

This repository focuses on:

- Exploratory data analysis (EDA)
- Feature scaling and class balancing
- Training and evaluating multiple classification models
- Comparing baseline ML models with a neural network approach

## Dataset Overview

- **Dataset name:** MAGIC Gamma Telescope (2004)
- **Source:** UCI Machine Learning Repository
- **DOI:** https://doi.org/10.24432/C52C8B
- **Primary files in this repo:**
  - `magic04.data` (raw tabular data)
  - `magic04.names` (dataset metadata and feature descriptions)
- **Number of instances:** 19,020
- **Number of attributes:** 11 total (10 input features + 1 class label)
- **Missing values:** None reported

### Class Labels

- `g` = gamma (signal)
- `h` = hadron (background)

In the notebook, labels are encoded to binary values:

- gamma (`g`) -> 1
- hadron (`h`) -> 0

## Feature Definitions

The 10 predictor features are Hillas-parameter style descriptors derived from telescope image geometry/intensity:

1. `fLength` - major axis of ellipse [mm]
2. `fWidth` - minor axis of ellipse [mm]
3. `fSize` - 10-log of sum of pixel contents [photons]
4. `fConc` - ratio of sum of two highest pixels over `fSize`
5. `fConc1` - ratio of highest pixel over `fSize`
6. `fAsym` - projected distance from highest pixel to center [mm]
7. `fM3Long` - cube root of third moment along major axis [mm]
8. `fM3Trans` - cube root of third moment along minor axis [mm]
9. `fAlpha` - angle of major axis with vector to origin [deg]
10. `fDist` - distance from origin to ellipse center [mm]

## Modeling Workflow

The notebook follows this pipeline:

1. Load data and assign column names.
2. Inspect class balance and basic distributions.
3. Encode class labels (`g/h` -> `1/0`).
4. Shuffle and split into train/validation/test (70%/15%/15%).
5. Standardize features using `StandardScaler`.
6. Address class imbalance on training data with `RandomOverSampler`.
7. Train and evaluate multiple models:
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Feedforward Neural Network (TensorFlow/Keras)
8. Report classification metrics (`precision`, `recall`, `f1-score`, support).

## Repository Structure

- `magic_gamma.ipynb` - main analysis and modeling notebook
- `magic04.data` - dataset values
- `magic04.names` - dataset documentation and attribute descriptions
- `gammaenv/` - local Python virtual environment

## Environment and Dependencies

The project uses Python with common data science and ML libraries, including:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- tensorflow / keras

## How to Run

1. Open this folder in VS Code or Jupyter.
2. Activate the existing virtual environment (optional):

```bash
source gammaenv/bin/activate
```

3. Launch Jupyter and open the notebook:

```bash
jupyter notebook
```

4. Run `magic_gamma.ipynb` cells in order.

## Notes on Evaluation

For this dataset, standard accuracy alone may be insufficient due to class-imbalance and asymmetric error costs (false positives on background can be expensive). Classification reports are included in the notebook, and ROC-based analysis is a good next step for model comparison under operating constraints.

## Citation

Bock, R. (2004). MAGIC Gamma Telescope [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B
