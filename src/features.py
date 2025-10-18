import pandas as pd

COL_NAMES = {
    "gender": "gender",
    "race/ethnicity": "race_ethnicity",
    "parental level of education": "parent_edu",
    "lunch": "lunch",
    "test preparation course": "test_prep",
    "math score": "math",
    "reading score": "reading",
    "writing score": "writing",
}

COLLEGE_WORDS = ["bachelor", "master", "associate", "college", "some college"]

def load_and_engineer(path):
    df = pd.read_csv(path)
    df = df.rename(columns=COL_NAMES)

    df["avg_score"] = df[["math", "reading", "writing"]].mean(axis=1)

    df["parent_college"] = df["parent_edu"].apply(
        lambda x: int(any(word in str(x).lower() for word in COLLEGE_WORDS))
    )

    def get_letter(avg):
        if avg >= 90: return "A"
        elif avg >= 80: return "B"
        elif avg >= 70: return "C"
        elif avg >= 60: return "D"
        else: return "F"

    df["letter_grade"] = df["avg_score"].apply(get_letter)

    return df

def split_features_target(df, target):
    features = [
        "gender", "race_ethnicity", "parent_edu", "lunch", "test_prep",
        "math", "reading", "writing", "avg_score", "parent_college"
    ]
    X = df[features]
    y = df[target]
    return X, y
