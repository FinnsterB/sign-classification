from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import get_all_features
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pdq
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import pickle

x, y = get_all_features("segmented_data")

features = np.array(x)
labels = np.array(y)  # Assuming two classes (100 and 50)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.4, random_state=42
)

# Choose a classifier (RandomForest in this case)
clf = RandomForestClassifier()

# Train the classifier if there's no pre-trained one available
if os.path.exists("clf.pkl"):
    with open("clf.pkl", "rb") as file:
        clf = pickle.load(file)
else:
    clf.fit(X_train, y_train)
    with open("clf.pkl", "wb") as file:
        pickle.dump(clf, file)

# Make predictions and evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using a heatmap for better visualization
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
plt.title("Confusion Matrix")
plt.show()

# Optional: Classification report for more detailed analysis
print("Classification Report:")
print(classification_report(y_test, y_pred))

clf_svc = SVC(kernel="linear")

# Train the classifier if there's no pre-trained one available
if os.path.exists("clf_svc.pkl"):
    with open("clf_svc.pkl", "rb") as file:
        clf_svc = pickle.load(file)
else:
    clf_svc.fit(X_train, y_train)
    with open("clf_svc.pkl", "wb") as file:
        pickle.dump(clf_svc, file)

y_pred_svc = clf_svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

conf_matrix = confusion_matrix(y_test, y_pred_svc)

# Display the confusion matrix using a heatmap for better visualization
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
plt.title("Confusion Matrix")
plt.show()

print("Classification Report")
print(classification_report(y_test, y_pred_svc))

clf_neighbor = KNeighborsClassifier(n_neighbors=5)

# Train the classifier if there's no pre-trained one available
if os.path.exists("clf_neighbor.pkl"):
    with open("clf_neighbor.pkl", "rb") as file:
        clf_neighbor = pickle.load(file)
else:
    clf_neighbor.fit(X_train, y_train)
    with open("clf_neighbor.pkl", "wb") as file:
        pickle.dump(clf_neighbor, file)


y_pred_neighbor = clf_neighbor.predict(X_test)
accuracy_neighbor = accuracy_score(y_test, y_pred_neighbor)

conf_matrix = confusion_matrix(y_test, y_pred_neighbor)

# Display the confusion matrix using a heatmap for better visualization
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
plt.title("Confusion Matrix")
plt.show()

print("Classification Report")
print(classification_report(y_test, y_pred_neighbor))
