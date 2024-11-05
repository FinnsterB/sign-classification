from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    learning_curve,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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

    report = classification_report(y_test, y_pred)
    with open(
        os.path.join(reports_dir, f"{model_name}_classification_report.txt"), "w"
    ) as f:
        f.write(report)





def plot_learning_curve(estimator, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42,
    )

    # Calculate the mean and standard deviation for train and test scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Store train sizes and test mean accuracy in the dictionary
    learning_curve_results[title] = (train_sizes, test_mean)

    # Original plotting and saving (optional for individual plots)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Test Score")

    plt.fill_between(
        train_sizes,
        train_mean - np.std(train_scores, axis=1),
        train_mean + np.std(train_scores, axis=1),
        color="r",
        alpha=0.2,
    )
    plt.fill_between(
        train_sizes,
        test_mean - np.std(test_scores, axis=1),
        test_mean + np.std(test_scores, axis=1),
        color="g",
        alpha=0.2,
    )

    plt.title(f"{title} Learning Curve (Accuracy)")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(learning_curve_dir, f"{title}_learning_curve.png"))
    plt.close()


# Define classifiers and their parameter grids for GridSearchCV
classifiers = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["minkowski", "euclidean"],
            "leaf_size": [30, 40, 50],
        },
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
        },
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"],
        },
    ),
    "Gaussian Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-8, 1e-7]}),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
        },
    ),
    # "SVC": (SVC(), {
    #     'C': [0.1, 1, 10],  # Reduced options to minimize grid search combinations
    #     'kernel': ['linear', 'rbf'],  # Removed 'poly' to avoid long training times
    #     'gamma': [0.1, 1, 'scale'],  # Replaced 'auto' with smaller gamma values for faster convergence
    #     'class_weight': [None, 'balanced']
    # }),
    # "GradientBoosting": (GradientBoostingClassifier(), {
    #     'n_estimators': [50, 100, 200],             # Number of boosting stages to perform
    #     'learning_rate': [0.01, 0.1, 0.2],          # Step size shrinkage used to prevent overfitting
    #     'max_depth': [3, 5, 7],                     # Maximum depth of the individual estimators
    #     'min_samples_split': [2, 5, 10],            # Minimum samples required to split an internal node
    #     'min_samples_leaf': [1, 2, 4],              # Minimum samples required at each leaf node
    #     'max_features': ['sqrt', 'log2', None],     # Number of features to consider when looking for the best split
    #     'subsample': [0.8, 1.0],                    # Fraction of samples used for fitting individual trees
    # })
    "Histogram-based Gradient Boosting": (
        HistGradientBoostingClassifier(),
        {
            "max_iter": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "min_samples_leaf": [1, 2, 4],
            "max_bins": [255, 511, 1023],  # Number of bins for histogram
        },
    ),
    "Bagging Classifier": (
        BaggingClassifier(random_state=42),
        {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.75, 1.0],
            "max_features": [0.5, 0.75, 1.0],
            "bootstrap": [True, False],
        },
    ),
}

def initClassifiers():
    for name, (clf, param_grid) in classifiers.items():
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                clf = pickle.load(file)
            print(f"Loaded pre-trained modeltest_scores for {name}.")
        else:
            print("Model not available")

def useClassifiers(x):
    y = {}
    for name, (clf, param_grid) in classifiers.items():
        x[name] = {clf.predict_proba(x)}
    
    return y
    
        



if __name__ == "__main__":


    # Dictionary to store classifier learning curve data
    learning_curve_results = {}

    # Load features and labels
    x, y = get_all_features("segmented_data")
    features = np.array(x)
    labels = np.array(y)

    # Split the data into training and testing sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=42
    )

    # Prepare directories for saving results
    output_dir = "data_classification"
    confusion_matrix_dir = os.path.join(output_dir, "confusion_matrices")
    reports_dir = os.path.join(output_dir, "classification_reports")
    models_dir = os.path.join(output_dir, "models")
    learning_curve_dir = os.path.join(output_dir, "learning_curves")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(learning_curve_dir, exist_ok=True)
    for name, (clf, param_grid) in classifiers.items():
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                clf = pickle.load(file)
            print(f"Loaded pre-trained modeltest_scores for {name}.")
        else:
            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            print(f"Best parameters for {name}: {grid_search.best_params_}")

            clf = grid_search.best_estimator_
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
            print(f"{name} is performing really good.")

        save_results(y_test, y_pred, name)

        # Plot learning curve for each classifier
        plot_learning_curve(clf, name, X_train, y_train)

    # Plot all classifiers' test accuracy on a combined graph
    plt.figure(figsize=(12, 8))
    for name, (train_sizes, test_mean) in learning_curve_results.items():
        plt.plot(train_sizes, test_mean, marker="o", label=f"{name} Test Accuracy")

    plt.title("Classifier Test Accuracy vs Training Samples")
    plt.xlabel("Training Samples")
    plt.ylabel("Test Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "all_classifiers_test_accuracy.png"))
    plt.show()

    print("\nAll results and models saved in the 'data_classification' folder.")