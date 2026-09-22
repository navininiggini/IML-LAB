# ICS1512: Machine Learning Algorithms Laboratory (IML-LAB)

Welcome to the **Machine Learning Algorithms Laboratory (`IML-LAB`)** repository. This workspace contains end-to-end implementations of classical and modern machine learning algorithms, exploratory data analysis pipelines, model evaluations, and academic lab reports.

**GitHub Repository:** [https://github.com/navininiggini/IML-LAB](https://github.com/navininiggini/IML-LAB)

---

## Repository Structure & File Context

Every directory is organized contextually such that all related notebooks, datasets, LaTeX reports, and figures reside strictly within their dedicated experiment folder:

```text
IML-LAB/
├── .venv/                                       # Dedicated Python 3.12 virtual environment
├── requirements.txt                             # Pinned Python package dependencies
├── test_all_experiments.py                     # Automated end-to-end verification test suite
├── README.md                                    # Repository documentation & context guide
│
├── Experiment-1/                                # Experiment 1: EDA & Preprocessing Pipelines
│   ├── Experiment_1_EDA.ipynb                   # Main notebook: Tabular, Text & Image pipelines
│   ├── exp1_initial.ipynb                       # Initial experimental reference notebook
│   ├── email.csv                                # Text classification dataset (SMS/Email Ham vs Spam)
│   ├── loan_data.csv                            # Tabular financial loan dataset
│   └── english.csv                              # Character recognition image dataset metadata
│
├── Experiment-2/                                # Experiment 2: Spam Classification (NB & KNN)
│   ├── Experiment_2_Spam_Classification.ipynb   # Gaussian/Multinomial/Bernoulli NB & KNN tuning
│   ├── spambase_csv.csv                         # Spambase dataset (57 features, binary labels)
│   ├── Report.tex                               # Formal LaTeX lab report
│   └── images/                                  # 13 generated figures (ROC, PR, CV curves, etc.)
│       ├── fig01_class_distribution.png
│       ├── fig02_correlation_heatmap.png
│       └── ... (13 plots)
│
├── Experiment-3/                                # Experiment 3: Regression Analysis
│   ├── Experiment_3_Regression_Analysis.ipynb   # Linear, Ridge, Lasso & Elastic Net regression
│   └── loan_data.csv                            # Loan Sanction dataset for target amount prediction
│
├── Experiment-4/                                # Experiment 4: Logistic Regression & SVM
│   ├── Experiment_4_LR_SVM.ipynb                # Linear & Kernel-based SVM + Logistic Regression
│   ├── spambase_csv.csv                         # Local copy of Spambase dataset
│   ├── Report.tex                               # Formal LaTeX lab report
│   └── images/                                  # 5 generated figures (kernel accuracy, confusion matrices)
│       ├── figure1.png
│       └── ... (5 plots)
│
├── Experiment-5/                                # Experiment 5: Decision Trees & Random Forests
│   ├── Experiment_5_DT_RF.ipynb                 # Comparative study on Wisconsin Breast Cancer dataset
│   ├── Report.tex                               # Formal LaTeX lab report
│   └── images/                                  # 6 generated figures (feature importance, ROC curves)
│       ├── figure1_class_distribution.png
│       └── ... (6 plots)
│
├── Experiment-6/                                # Experiment 6: Dimensionality Reduction (PCA)
│   ├── Experiment_6_PCA_Model_Evaluation.ipynb  # 10 classifiers evaluated with/without PCA
│   ├── Report.tex                               # Formal LaTeX lab report with all tables & Q&A
│   ├── outputs/                                 # Exported CSV tables (Table 1 to Table 5)
│   └── images/                                  # 6 generated figures (Scree plot, stability, ROC)
│
├── Experiment-7/                                # Experiment 7: Ensemble Learning
│   ├── Experiment_7_Ensemble_Learning.ipynb     # Bagging, AdaBoost, Gradient Boosting & Stacking
│   ├── Report.tex                               # Formal LaTeX lab report with Tables 1-4 & Q&A
│   ├── outputs/                                 # Exported CSV tables (Table 1 to Table 4)
│   └── images/                                  # 6 generated figures (bias-variance, tuning, ROC)
│
├── Experiment-8/                                # Experiment 8: Clustering HAR Data
│   ├── Experiment_8_Clustering_HAR.ipynb        # K-Means, DBSCAN & Hierarchical Clustering
│   ├── Report.tex                               # Formal LaTeX lab report with Tables 1-3 & Q&A
│   ├── data/                                    # Processed HAR dataset files
│   ├── outputs/                                 # Exported CSV tables (Elbow results, DBSCAN, metrics)
│   └── images/                                  # 6 generated figures (Elbow, t-SNE, dendrograms)
│
└── Experiment-9/                                # Experiment 9: Perceptron vs MLP (A/B Experiment)
    ├── Experiment_9_Perceptron_vs_MLP.ipynb     # Single-Layer PLA from scratch vs Tuned Deep MLP
    ├── Report.tex                               # Formal LaTeX lab report with Tables 1-2 & Q&A
    ├── data/                                    # English handwritten characters & metadata
    ├── outputs/                                 # Exported CSV tables (tuning grid & A/B comparison)
    └── images/                                  # 6 generated figures (loss curves, ROC, super-class CM)
```

---

## Detailed Experiment Contexts

### [Experiment-1: Exploratory Data Analysis & Preprocessing Pipelines](file:///home/jarvis/Desktop/IML-LAB/Experiment-1)
- **Objective:** Build unified preprocessing pipelines for tabular, text, and image datasets.
- **Components:**
  - **Tabular Pipeline:** Handles missing values, encodes categorical features with `LabelEncoder`, scales features with `StandardScaler`, and performs feature selection with `SelectKBest`.
  - **Text Pipeline:** Cleans text, tokenizes, removes stopwords via NLTK, and computes TF-IDF representations.
  - **Image Pipeline:** Loads pixel arrays, normalizes values, and prepares stratified train-validation-test splits.
- **Datasets:** `email.csv`, `loan_data.csv`, `english.csv`.

### [Experiment-2: Email Spam/Ham Classification using Naïve Bayes and KNN](file:///home/jarvis/Desktop/IML-LAB/Experiment-2)
- **Objective:** Compare probabilistic classifiers (GaussianNB, MultinomialNB, BernoulliNB) with distance-based K-Nearest Neighbors (KNN).
- **Techniques:**
  - Hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`.
  - Comparison of nearest-neighbor search algorithms: `KDTree` vs. `BallTree`.
  - Computational timing benchmarks for training and inference.
  - Evaluation via 5-Fold Cross Validation, ROC curves, and Precision-Recall curves.
- **Dataset:** `spambase_csv.csv`.

### [Experiment-3: Regression Analysis](file:///home/jarvis/Desktop/IML-LAB/Experiment-3)
- **Objective:** Predict loan sanction amounts using regularized linear regression techniques.
- **Models:**
  - Ordinary Least Squares (OLS) Linear Regression.
  - Ridge Regression ($L_2$ regularization) with $\alpha$ hyperparameter tuning.
  - Lasso Regression ($L_1$ regularization / sparse feature selection) with $\alpha$ tuning.
  - Elastic Net (combined $L_1 + L_2$) with $\alpha$ and $l_1$-ratio optimization.
- **Evaluation:** MAE, MSE, RMSE, $R^2$, residual plots, bias-variance analysis, and coefficient shrinkages.
- **Dataset:** `loan_data.csv`.

### [Experiment-4: Binary Classification using Linear and Kernel-Based Models](file:///home/jarvis/Desktop/IML-LAB/Experiment-4)
- **Objective:** Classify email content using Logistic Regression and Support Vector Machines (SVM).
- **Techniques:**
  - Logistic Regression regularizer tuning (L1, L2, ElasticNet).
  - SVM kernel comparison (`linear`, `rbf`, `poly`, `sigmoid`).
  - Grid search over penalty parameter $C$, kernel coefficient $\gamma$, and polynomial degree.
- **Dataset:** `spambase_csv.csv`.

### [Experiment-5: Decision Tree and Random Forest - A Comparative Study](file:///home/jarvis/Desktop/IML-LAB/Experiment-5)
- **Objective:** Evaluate how ensembling affects overfitting, bias, and variance when transitioning from a single Decision Tree to a Random Forest ensemble.
- **Techniques:**
  - Hyperparameter tuning across `criterion` (Gini, Entropy), `max_depth`, `n_estimators`, and `bootstrap`.
  - Feature importance ranking for clinical breast cancer diagnostics.
  - Confusion matrix analysis and Stratified 5-Fold Cross-Validation.
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset (scikit-learn `load_breast_cancer()`).

### [Experiment-6: Dimensionality Reduction and Model Evaluation (With and Without PCA)](file:///home/jarvis/Desktop/IML-LAB/Experiment-6)
- **Objective:** Quantify the effect of Principal Component Analysis on 10 diverse classifiers (SVM, Naïve Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking).
- **Techniques:**
  - PCA eigenvalue decomposition, Scree plot, and 95% cumulative variance retention (10 components).
  - Grid search hyperparameter tuning under both No-PCA and With-PCA conditions.
  - Fold-wise Stratified 5-Fold Cross-Validation variance and stability analysis.
  - Rigorous analysis of observation questions regarding axis-aligned tree splits vs. linear boundary rotations.
- **Outputs:** Complete formal LaTeX report (`Report.tex`), 6 publication figures, and Table 1 to Table 5 CSV datasets.
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset.

### [Experiment-7: Ensemble Learning (Bagging, Boosting, Stacking)](file:///home/jarvis/Desktop/IML-LAB/Experiment-7)
- **Objective:** Evaluate individual base learners against three ensemble paradigms (Bagging with Decision Trees, AdaBoost / Gradient Boosting, and heterogeneous Stacking) on clinical diagnostic classification.
- **Techniques:**
  - Bagging hyperparameter tuning over ensemble size ($B \in [5, 150]$) and bootstrap sample ratios.
  - Boosting grid search over learning rates ($\eta \in [0.01, 1.0]$) and base estimator depths.
  - Stacking Classifier architecture: Diverse base learners (Decision Tree, Logistic Regression, GaussianNB, SVM) combined via cross-validated out-of-fold meta-features.
  - Empirical bias-variance decomposition confirming variance mitigation via bagging and sequential bias reduction via boosting.
  - ROC-AUC evaluation, confusion matrix diagnostics, and feature importance rankings.
- **Outputs:** Complete formal LaTeX report (`Report.tex`), 6 publication figures, and Table 1 to Table 4 CSV datasets.
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset (`load_breast_cancer()`, 569 samples, 30 features).

### [Experiment-8: Clustering Human Activity Recognition Data (K-Means, DBSCAN, HAC)](file:///home/jarvis/Desktop/IML-LAB/Experiment-8)
- **Objective:** Perform unsupervised discovery of physical human activity patterns from smartphone inertial sensor telemetry using centroid-based, density-based, and connectivity-based clustering.
- **Techniques:**
  - Standardized 561-feature time/frequency domain accelerometer and gyroscope telemetry.
  - K-Means clustering with Elbow Method (inertia / WCSS) and Silhouette analysis across $k=2 \dots 10$.
  - DBSCAN density clustering with $k$-distance graph knee point estimation ($\epsilon \in [5.0, 25.0]$) analyzing high-dimensional density disparities.
  - Hierarchical Agglomerative Clustering (HAC) using Ward's minimum variance criterion and full dendrogram topology visualization.
  - Dual evaluation: Internal metrics (Silhouette Score, Davies-Bouldin, Calinski-Harabasz) and External ground-truth metrics (Adjusted Rand Index, Normalized Mutual Information).
  - 2D Manifold Visualizations: Global variance PCA and non-linear t-SNE projection maps.
- **Outputs:** Complete formal LaTeX report (`Report.tex`), 6 publication figures, and Table 1 to Table 3 CSV datasets.
- **Dataset:** UCI Human Activity Recognition Using Smartphones (10,299 samples, 561 features).

### [Experiment-9: Perceptron vs Multilayer Perceptron (A/B Experiment)](file:///home/jarvis/Desktop/IML-LAB/Experiment-9)
- **Objective:** Conduct a formal A/B comparative experiment testing the linear separability hypothesis: contrasting a single-layer Perceptron Learning Algorithm (PLA) written from scratch against a deep Multilayer Perceptron (MLP) on 62-class handwritten character recognition.
- **Techniques:**
  - Optical character data pipeline: 3,410 handwritten character images standardized to $32 \times 32$ grayscale (1,024 input features).
  - Condition A (PLA): Vectorized One-vs-Rest Perceptron implemented from scratch with step activation and perceptron weight update rule ($w \leftarrow w + \eta (y_i - \hat{y}_i) x_i$).
  - Condition B (MLP): Multi-layer feedforward network with ReLU non-linear activations, cross-entropy backpropagation, and Adam optimization.
  - Hyperparameter tuning: Architectural grid search across hidden layer configurations (`(100,)`, `(256, 128)`, `(512, 256, 128)`), learning rates, and $L_2$ regularization penalties ($\alpha$).
  - Visualizations: Training loss convergence curves, One-vs-Rest ROC curves, 2D PCA decision boundary comparison, and super-class confusion matrices (Digits, Uppercase, Lowercase).
- **Outputs:** Complete formal LaTeX report (`Report.tex`), 6 publication figures, and Table 1 to Table 2 CSV datasets.
- **Dataset:** English Handwritten Characters Dataset (3,410 images across 62 classes).


---

## Environment Setup & Installation

A dedicated virtual environment (`.venv`) has been pre-configured in this repository using **Python 3.12**.

### 1. Activate the Virtual Environment

```bash
cd /home/jarvis/Desktop/IML-LAB
source .venv/bin/activate
```

### 2. (Optional) Re-install Dependencies

If recreating the environment from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Automated Verification Suite

To verify that all experiment notebooks, datasets, and algorithms run end-to-end without errors:

```bash
source .venv/bin/activate
python test_all_experiments.py
```

### Verification Output:
```text
======================================================================
           ICS1512 - Machine Learning Algorithms Laboratory
                 Automated Notebook Verification Suite
======================================================================

[1/9] Testing: Experiment 1: Exploratory Data Analysis & Preprocessing
      File:    Experiment-1/Experiment_1_EDA.ipynb
      Result:  PASSED (5.1s)

[2/9] Testing: Experiment 2: Email Spam/Ham Classification (Naïve Bayes & KNN)
      File:    Experiment-2/Experiment_2_Spam_Classification.ipynb
      Result:  PASSED (16.7s)

[3/9] Testing: Experiment 3: Regression Analysis (Linear, Ridge, Lasso, Elastic Net)
      File:    Experiment-3/Experiment_3_Regression_Analysis.ipynb
      Result:  PASSED (6.2s)

[4/9] Testing: Experiment 4: Binary Classification (Logistic Regression & SVM)
      File:    Experiment-4/Experiment_4_LR_SVM.ipynb
      Result:  PASSED (41.4s)

[5/9] Testing: Experiment 5: Comparative Classification (Decision Tree & Random Forest)
      File:    Experiment-5/Experiment_5_DT_RF.ipynb
      Result:  PASSED (118.6s)

[6/9] Testing: Experiment 6: Dimensionality Reduction and Model Evaluation (With and Without PCA)
      File:    Experiment-6/Experiment_6_PCA_Model_Evaluation.ipynb
      Result:  PASSED (31.8s)

[7/9] Testing: Experiment 7: Bagging, Boosting, and Stacked Ensemble Models
      File:    Experiment-7/Experiment_7_Ensemble_Learning.ipynb
      Result:  PASSED (48.3s)

[8/9] Testing: Experiment 8: Clustering Human Activity Recognition Data (K-Means, DBSCAN, HAC)
      File:    Experiment-8/Experiment_8_Clustering_HAR.ipynb
      Result:  PASSED (22.5s)

[9/9] Testing: Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment)
      File:    Experiment-9/Experiment_9_Perceptron_vs_MLP.ipynb
      Result:  PASSED (38.1s)

======================================================================
                       VERIFICATION SUMMARY
======================================================================
 - [PASSED] Experiment 1: Exploratory Data Analysis & Preprocessing
 - [PASSED] Experiment 2: Email Spam/Ham Classification (Naïve Bayes & KNN)
 - [PASSED] Experiment 3: Regression Analysis (Linear, Ridge, Lasso, Elastic Net)
 - [PASSED] Experiment 4: Binary Classification (Logistic Regression & SVM)
 - [PASSED] Experiment 5: Comparative Classification (Decision Tree & Random Forest)
 - [PASSED] Experiment 6: Dimensionality Reduction and Model Evaluation (With and Without PCA)
 - [PASSED] Experiment 7: Bagging, Boosting, and Stacked Ensemble Models
 - [PASSED] Experiment 8: Clustering Human Activity Recognition Data (K-Means, DBSCAN, HAC)
 - [PASSED] Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment)
----------------------------------------------------------------------
Total: 9 | Passed: 9 | Failed: 0
======================================================================
```

---

## Running in Jupyter

To interactively view, modify, or execute any experiment in JupyterLab:

```bash
source .venv/bin/activate
jupyter lab
```
Select the **`Python (IML-LAB)`** kernel from the dropdown.
