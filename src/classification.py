from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import get_all_features
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

x, y = get_all_features("segmented_data")

features = np.array(x)
labels = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

output_dir = 'data_classification'
os.makedirs(output_dir, exist_ok=True)

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
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

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
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

    save_results(y_test, y_pred, name)

print("All results saved in the 'data_classification' folder.")
