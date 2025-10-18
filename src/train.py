import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from .constants import RANDOM_STATE, DATA_PATH, TARGET_LABEL, RESULTS_DIR
from .features import load_and_engineer, split_features_target
from .evaluation import save_confusion_matrix, get_metrics

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_and_engineer(DATA_PATH)
    X, y = split_features_target(df, TARGET_LABEL)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cat_cols = ["gender", "race_ethnicity", "parent_edu", "lunch", "test_prep"]
    num_cols = ["math", "reading", "writing", "avg_score", "parent_college"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    models = {
        "Baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "k-NN": KNeighborsClassifier(n_neighbors=10)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    labels = sorted(y.unique())

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro").mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        scores = get_metrics(y_test, y_pred, name)
        scores["Mean"] = round(cv_score, 5)
        results.append(scores)

        out_file = os.path.join(RESULTS_DIR, f"confusion_{name.replace(' ', '_')}.png")
        save_confusion_matrix(y_test, y_pred, labels, out_file, f"{name} Confusion Matrix")

        print(f"{name}: CV F1 = {cv_score:.3f}, Accuracy = {scores['Accuracy']:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    print("\n=== Final Results ===")
    print(results_df.sort_values("F1 Score", ascending=False))

if __name__ == "__main__":
    main()
