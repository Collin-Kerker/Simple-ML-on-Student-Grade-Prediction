# Student Grade Classifier (Shallow ML)

**Goal:** Predict a student's *letter grade* (A/B/C/D/F) using shallow learning techniques from study context and exam scores.

This repository is structured to satisfy the midterm specification you provided: baseline vs. shallow models, proper evaluation, reproducibility, and a short video/walkthrough.

## Dataset

- **Primary:** *Students Performance in Exams* — Kaggle (SPScientist, CC0)
- Local copy saved at `data/StudentsPerformance.csv`.
- Columns (original): `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`, `math score`, `reading score`, `writing score`.
- Engineered:
  - `avg_score = (math + reading + writing)/3`
  - `parent_college = 1 if parental education indicates any college/associate/bachelor/master else 0`
  - `letter_grade` from `avg_score` using standard 10-point bins: A (90+), B (80-89), C (70-79), D (60-69), F (<60).

## Quickstart

```bash
# (optional) create/activate a venv first
pip install -r requirements.txt

# Run the training script end-to-end
python -m src.train
```

Outputs (confusion matrices, metrics tables) are written to `results/` and also printed.
A fully reproducible EDA + modeling notebook is in `notebooks/midterm_pipeline.ipynb`.

## Reproducibility

- Random seeds fixed (`RANDOM_STATE=42`).
- Versions pinned in `requirements.txt`.
- Script reads `data/StudentsPerformance.csv`. If missing, update `DATA_PATH` in `src/constants.py`.

## Repository Layout

```
student-grade-classifier/
├── README.md
├── requirements.txt
├── data/
│   └── StudentsPerformance.csv
├── notebooks/
│   └── midterm_pipeline.ipynb
├── src/
│   ├── constants.py
│   ├── features.py
│   ├── evaluation.py
│   └── train.py
├── results/
└── presentation/
```

## How to Reproduce Results

1. Install dependencies and run `python -m src.train`.
2. Check `results/` for figures and metrics CSV.
3. Optionally open the notebook for EDA and re-run cells.

## Notes

- Baseline: Majority-class classifier.
- Models: Logistic Regression, Decision Tree, k-NN.
- Metrics: Accuracy, macro-F1; confusion matrices saved as PNG.
- Split: Stratified train/test split (80/20) + cross-validation (5-fold) on train.
