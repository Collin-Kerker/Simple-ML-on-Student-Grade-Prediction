import os

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "KaggleStudentsData.csv")
TARGET_LABEL = "letter_grade"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")