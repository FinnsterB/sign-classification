import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, learning_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
from feature_extraction import get_all_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

x, y = get_all_features("segmented_data")
features = np.array(x)
labels = np.array(y)

X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

X_valid, X_train, Y_valid, Y_train = train_test_split(
    X_train_val, Y_train_val, test_size=0.75, random_state=42
)

output_dir = 'data_classification'
os.makedirs(output_dir, exist_ok=True)

confusion_matrix_dir = os.path.join(output_dir, 'confusion_matrices')
reports_dir = os.path.join(output_dir, 'classification_reports')
learning_curve_dir = os.path.join(output_dir, 'learning_curves')
os.makedirs(confusion_matrix_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(learning_curve_dir, exist_ok=True)

def save_results(y_test, y_pred, model_name):
    # Generate and save confusion matrix
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
    plt.savefig(os.path.join(confusion_matrix_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    report = classification_report(y_test, y_pred)
    with open(os.path.join(reports_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

def plot_learning_curve(estimator, X, y, model_name, ylim=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 10)):
    """Generates and saves a learning curve for the given model."""
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curves ({model_name})")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        return_times=True
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    # Save the plot
    filename = os.path.join(learning_curve_dir, f"{model_name}_learning_curve.png")
    plt.savefig(filename)
    print(f"Learning curve plot saved as: {filename}")
    plt.close()

classifiers = {
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(kernel="linear"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gaussian Naive Bayes": GaussianNB(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100)
}

for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_valid, Y_valid, cv=5)

    print(f"{name} Cross-Validation Scores (Validation Set): {cv_scores}")
    print(f"{name} Mean Cross-Validation Accuracy (Validation Set): {np.mean(cv_scores):.4f}")

    clf.fit(X_valid, Y_valid)

    y_pred = clf.predict(X_test)

    accuracy_validation = accuracy_score(Y_valid, clf.predict(X_valid))
    accuracy_test = accuracy_score(Y_test, y_pred)

    precision = precision_score(Y_test, y_pred, average='weighted')
    recall = recall_score(Y_test, y_pred, average='weighted')
    f1 = f1_score(Y_test, y_pred, average='weighted')

    print(f"{name} Validation Accuracy: {accuracy_validation:.4f}")
    print(f"{name} Test Accuracy: {accuracy_test:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} F1-score: {f1:.4f}")

    plot_learning_curve(clf, X_valid, Y_valid, name)

    save_results(Y_test, y_pred, name)

print("All results and learning curves saved in the 'data_classification' folder.")
