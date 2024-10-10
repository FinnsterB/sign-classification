import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import os
import cv2  # For reading images
from pathlib import Path  # For navigating file paths


# Define a function to load your custom dataset
def load_data_from_directory(data_directory):
    image_data = []
    labels = []

    # Get all subdirectories (these are the labels)
    data_path = Path(data_directory)
    subfolders = [f for f in data_path.iterdir() if f.is_dir()]

    for subfolder in subfolders:
        # The label will be the name of the subfolder
        label = subfolder.name
        # Get all .png files in the subfolder
        for image_path in subfolder.glob("*.png"):
            # Read the image
            image = cv2.imread(str(image_path))
            if image is not None:
                # Convert to grayscale (if necessary)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Resize to a fixed size (if needed, e.g., 64x64 pixels)
                resized_image = cv2.resize(gray_image, (64, 64))
                # Flatten the image to turn it into a feature vector
                flattened_image = resized_image.flatten()
                # Add to the dataset
                image_data.append(flattened_image)
                labels.append(label)

    # Convert lists to NumPy arrays
    X = np.array(image_data)
    y = np.array(labels)

    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.4, random_state=42)


def create_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=0.5, random_state=42))
    ])


def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.set_title("Learning Curves (SVM, Polynomial Kernel)")
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt


def save_plot(plt):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    filename = f"{script_name}_learning_curve.png"
    plt.savefig(filename)
    print(f"Learning curve plot saved as: {filename}")


def main():
    data_directory = r'C:\Users\Blast\Desktop\Machine Learning\segmented_dataset'  # Update this to your folder
    X_train, X_test, y_train, y_test = load_data_from_directory(data_directory)
    clf = create_classifier()
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    plt = plot_learning_curve(clf, X_train, y_train, cv=cv, n_jobs=-1)
    save_plot(plt)
    plt.show()


if __name__ == "__main__":
    main()
