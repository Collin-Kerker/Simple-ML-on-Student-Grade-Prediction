# Student Grade Prediction

A comparative machine learning project that predicts student letter grades based on demographic and academic factors using multiple classification algorithms.

## Project Overview

This project explores how various socioeconomic and educational factors influence student academic performance. By analyzing factors such as parental education level, lunch program participation, test preparation, and prior test scores, the model predicts letter grades (A, B, C, D, F) and identifies key indicators of student success.

## Motivation

Understanding the factors that influence student performance can help educators provide targeted support to students who need it most. This project aims to:
- Identify students who may benefit from additional academic support
- Understand how socioeconomic factors (like free/reduced lunch status) correlate with academic outcomes
- Demonstrate the impact of test preparation and parental education on student success
- Provide data-driven insights for educational interventions

## Dataset

**Source:** Kaggle Students Performance Dataset  
**Size:** 1,000 student records  
**Features:**
- **Demographic:** Gender, race/ethnicity
- **Socioeconomic:** Parental education level, lunch type (standard vs. free/reduced)
- **Academic:** Math score, reading score, writing score, test preparation course completion
- **Target Variable:** Letter grade (A, B, C, D, F) - derived from test score averages

## Models Implemented

This project compares four different classification approaches:

1. **Baseline (Majority Classifier)** - Predicts most common class for comparison
2. **Logistic Regression** - Linear classification model
3. **Decision Tree** - Tree-based interpretable classifier
4. **k-Nearest Neighbors (k-NN)** - Instance-based learning with k=10

## Results

| Model | Cross-Validation F1 (Macro) | Test Accuracy | Test F1 (Macro) |
|-------|----------------------------|---------------|-----------------|
| **Logistic Regression** | **0.918** | **94.5%** | **0.937** |
| Decision Tree | 0.884 | 91.5% | 0.912 |
| k-NN (k=10) | 0.771 | 85.5% | 0.853 |
| Baseline (Majority) | 0.089 | 28.5% | 0.089 |

**Best Model:** Logistic Regression achieved the highest performance with 94.5% accuracy and an F1 score of 0.937, significantly outperforming the baseline.

## Key Findings

1. **Test Preparation Matters:** Students who completed test preparation courses consistently achieved higher grades
2. **Socioeconomic Impact:** Lunch type (standard vs. free/reduced) showed correlation with performance, indicating that economic factors affect student outcomes
3. **Parental Education:** Higher parental education levels correlated with better student performance
4. **Score Patterns:** Math, reading, and writing scores were strong predictors of overall letter grades

These findings suggest that both academic support (test prep) and addressing socioeconomic barriers could improve student outcomes.

## Technologies Used

- **Language:** Python 3.13
- **Libraries:**
  - scikit-learn (modeling, preprocessing, evaluation)
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib (visualization)
- **Environment:** Jupyter Notebook, VS Code

## Setup & Installation

### Prerequisites
- Python 3.13 (or 3.7+)
- pip package manager

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/Collin-Kerker/Simple-ML-on-Student-Grade-Prediction.git
cd Simple-ML-on-Student-Grade-Prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy
pandas
matplotlib
scikit-learn
```

## Running the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks_and_results/Simple_ML_Notebook.ipynb
```
Run all cells to see data exploration, model training, and results.

### Option 2: Python Script
```bash
python -m src.train
```
Results will be saved to the `results/` folder.

## Reproducing Results

The random seed is set to `42` for reproducibility. Simply follow the installation steps above and run the notebook or training script - you should get identical results.

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
```

## Model Pipeline

The project uses scikit-learn pipelines with the following preprocessing:

1. **Categorical Features** (gender, race/ethnicity, parental education, lunch type, test prep):
   - One-Hot Encoding with unknown value handling

2. **Numerical Features** (math score, reading score, writing score, parental college indicator):
   - Standard Scaling (z-score normalization)

3. **Evaluation:**
   - 5-fold Stratified Cross-Validation
   - 80/20 train/test split
   - Metrics: Accuracy, F1 Score (macro), Confusion Matrix

## Project Structure

```
Simple-ML-on-Student-Grade-Prediction/
├── data/
│   └── KaggleStudentsData.csv            # Raw dataset (1,000 rows)
├── src/
│   ├── constants.py                      # Paths, random seed, target label
│   ├── features.py                       # Loading, feature engineering, letter-grade target
│   ├── evaluation.py                     # Metrics and confusion-matrix plots
│   └── train.py                          # Trains and evaluates all four models
├── notebooks_and_results/
│   └── Simple_ML_Notebook.ipynb          # Exploration and full analysis
├── results/                              # Confusion matrices + metrics_summary.csv
├── requirements.txt
└── README.md
```

## Future Improvements

- Implement ensemble methods (Random Forest, Gradient Boosting)
- Perform hyperparameter tuning with GridSearchCV
- Add feature importance analysis and visualization
- Explore regression models to predict exact scores
- Test on larger, more diverse datasets
- Create interactive dashboard for predictions

## What I Learned

- Comparative analysis of classification algorithms
- Importance of proper data preprocessing (encoding, scaling)
- Using scikit-learn pipelines for clean, reproducible ML workflows
- Model evaluation with cross-validation and multiple metrics
- Understanding how socioeconomic factors influence educational outcomes
- Practical application of statistical learning concepts

## Context

Midterm project for EGR 422 at California Baptist University.

## Author

**Collin Kerker**  
Computer Science Student @ California Baptist University  
[GitHub](https://github.com/Collin-Kerker) | [LinkedIn](https://www.linkedin.com/in/collin-kerker/)
