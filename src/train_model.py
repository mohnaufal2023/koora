import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_dataset(filename):
    path = os.path.join(FEATURE_DIR, filename)
    data = pd.read_csv(path, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

    return y_pred


def main():
    # Load dataset
    X_train, y_train = load_dataset("train_features.csv")
    X_test, y_test = load_dataset("test_features.csv")

    # Model individual
    svm = SVC(kernel="rbf", probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model individual
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # ==============================
    # SIMPAN MODEL TERBAIK (RF)
    # ==============================
    joblib.dump(rf, MODEL_PATH)

    # Evaluasi model individual
    evaluate_model(svm, X_test, y_test, "SVM")
    evaluate_model(knn, X_test, y_test, "KNN")
    evaluate_model(rf, X_test, y_test, "Random Forest")

    # Ensemble Learning - Hard Voting
    ensemble = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("knn", knn),
            ("rf", rf),
        ],
        voting="hard"
    )

    ensemble.fit(X_train, y_train)

    # Evaluasi ensemble
    y_pred_ensemble = evaluate_model(ensemble, X_test, y_test, "Ensemble (Hard Voting)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print("\n=== Confusion Matrix (Ensemble) ===")
    print(cm)

    # Classification Report
    print("\n=== Classification Report (Ensemble) ===")
    print(classification_report(y_test, y_pred_ensemble))


if __name__ == "__main__":
    main()
