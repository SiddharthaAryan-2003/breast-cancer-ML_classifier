# 🩺 Breast Cancer Prediction — Multi-Model ML Approach

A machine learning project that predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** using the Wisconsin Breast Cancer dataset. Multiple classification models are compared to find the best performer, which is then fine-tuned via hyperparameter optimization for maximum accuracy.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **File:** `Cancer_Data.csv`
- **Samples:** 569
- **Features:** 30 real-valued features computed from digitized images of fine needle aspirates (FNA) of breast masses
- **Target:** `diagnosis` — Malignant (M) or Benign (B)

### Feature Groups

| Group | Features |
|-------|----------|
| **Mean** | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| **SE** | Standard error of the above 10 features |
| **Worst** | Worst (largest) values of the above 10 features |

---

## 🏗️ Project Structure

```
br_cance/
├── Cancer_Data.csv                        # Dataset
├── Breast_Cancer_Prediction_test.ipynb    # Jupyter notebook (step-by-step)
├── breast_cancer_prediction.py            # Full Python script (multi-model)
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/breast-cancer-prediction.git
cd breast-cancer-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Script

```bash
python breast_cancer_prediction.py
```

### Run the Notebook

```bash
jupyter notebook Breast_Cancer_Prediction_test.ipynb
```

---

## 🔬 Methodology

### Step-by-Step Pipeline

1. **Load Data** — Read `Cancer_Data.csv` into a pandas DataFrame
2. **Explore Data** — Inspect shape, types, distributions, and missing values
3. **Clean Data** — Drop `id` and `Unnamed: 32` columns; handle NaNs
4. **Visualize Data** — Diagnosis distribution, correlation heatmap, box plots, pair plots
5. **Preprocess Data** — Label encode target, train/test split (80/20), StandardScaler
6. **Train Models** — Fit 10 different classifiers with 10-fold stratified cross-validation
7. **Evaluate Models** — Compare accuracy, F1-score, AUC-ROC; plot ROC curves
8. **Hyperparameter Tuning** — GridSearchCV on top 3 models
9. **Ensemble** — Soft voting classifier combining the best tuned models
10. **Final Summary** — Rank all models and report the overall best

---

## 🤖 Models Compared

| # | Model | Description |
|---|-------|-------------|
| 1 | Logistic Regression | Linear classifier with regularization |
| 2 | Random Forest | Ensemble of decision trees (bagging) |
| 3 | Support Vector Machine | Maximum-margin classifier with RBF kernel |
| 4 | K-Nearest Neighbors | Instance-based lazy learner |
| 5 | Decision Tree | Single tree classifier |
| 6 | Gradient Boosting | Sequential ensemble (boosting) |
| 7 | AdaBoost | Adaptive boosting |
| 8 | Extra Trees | Extremely randomized trees |
| 9 | Naive Bayes | Probabilistic Gaussian classifier |
| 10 | Neural Network (MLP) | Multi-layer perceptron |

---

## 📈 Results

### Model Comparison (Before Tuning)

| Model | CV Accuracy | Test Accuracy | AUC-ROC |
|-------|-------------|---------------|---------|
| Neural Network (MLP) | 0.9758 | **0.9825** | 0.9950 |
| Logistic Regression | 0.9758 | 0.9649 | 0.9960 |
| AdaBoost | 0.9735 | 0.9737 | 0.9854 |
| SVM | 0.9714 | 0.9737 | 0.9947 |
| Random Forest | 0.9626 | 0.9737 | 0.9929 |
| Extra Trees | 0.9670 | 0.9737 | 0.9988 |
| Gradient Boosting | 0.9671 | 0.9649 | 0.9947 |
| KNN | 0.9647 | 0.9561 | 0.9823 |
| Naive Bayes | 0.9361 | 0.9211 | 0.9891 |
| Decision Tree | 0.9166 | 0.9298 | 0.9246 |

> **Best Model:** Neural Network (MLP) — **98.25% test accuracy**

### Visualizations Generated

- `01_diagnosis_distribution.png` — Class balance bar & pie charts
- `02_correlation_heatmap.png` — Feature correlation matrix
- `03_boxplots.png` — Key features by diagnosis
- `04_pairplot.png` — Pairwise feature relationships
- `05_model_comparison.png` — CV vs Test accuracy bar chart
- `06_roc_curves.png` — ROC curves for all models
- `07_confusion_matrix_best.png` — Best model confusion matrix
- `08_confusion_matrix_tuned.png` — Tuned model confusion matrix

---

## 🛠️ Tech Stack

- **Python 3.12**
- **pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib & Seaborn** — Visualization
- **scikit-learn** — ML models, preprocessing, evaluation

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) for the dataset
- [scikit-learn documentation](https://scikit-learn.org/) for ML algorithms
