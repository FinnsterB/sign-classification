from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import get_all_features
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

x, y = get_all_features("segmented_data")
features = np.array(x)
labels = np.array(y)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.25, random_state=42
)

output_dir = "data_classification"
os.makedirs(output_dir, exist_ok=True)

confusion_matrix_dir = os.path.join(output_dir, "confusion_matrices")
reports_dir = os.path.join(output_dir, "classification_reports")
os.makedirs(confusion_matrix_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

models_dir = os.path.join(output_dir, "models")
os.makedirs(models_dir, exist_ok=True)


def save_results(y_test, y_pred, model_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(labels),
        yticklabels=np.unique(labels),
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(
        os.path.join(confusion_matrix_dir, f"{model_name}_confusion_matrix.png")
    )
    plt.close()

    # Save classification report to a text file
    report = classification_report(y_test, y_pred)
    with open(
        os.path.join(reports_dir, f"{model_name}_classification_report.txt"), "w"
    ) as f:
        f.write(report)


# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gaussian Naive Bayes": GaussianNB(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100),
}

for name, clf in classifiers.items():
    model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            clf = pickle.load(file)
        print(f"Loaded pre-trained model for {name}.")
    else:
        cv_scores = cross_val_score(clf, features, labels, cv=5)
        print(f"{name} Cross-Validation Scores: {cv_scores}")
        print(f"{name} Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")
        clf.fit(X_train, y_train)
        with open(model_path, "wb") as file:
            pickle.dump(clf, file)
        print(f"Trained and saved model for {name}.")

    y_pred = clf.predict(X_test)

    accuracy_train = accuracy_score(y_train, clf.predict(X_train))
    accuracy_test = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{name} Performance:")
    print(f"Training Accuracy: {accuracy_train:.4f}")
    print(f"Test Accuracy: {accuracy_test:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    if accuracy_train < 0.6 and accuracy_test < 0.6:
        print(f"{name} is likely underfitting.")
    elif accuracy_train > accuracy_test + 0.1:
        print(f"{name} is likely overfitting.")
    else:
        print(f"{name} is performing reasonably well.")

    save_results(y_test, y_pred, name)

print("\nAll results and models saved in the 'data_classification' folder.")
