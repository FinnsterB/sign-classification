import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from feature_extraction import get_all_features

FEATURES = [
    "Number of digits",
    "Perimeter",
    "Nr of circles",
    "Nr of unkown shapes",
    "Total nr of shapes",
]

FEATURE_DATA_FOLDER = "feature_plots"
os.makedirs(FEATURE_DATA_FOLDER, exist_ok=True)


def features_to_dataframe(x, y):
    df = pd.DataFrame(
        x,
        columns=FEATURES,
    )
    df["Label"] = y
    return df


def histogram(df, feature_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature_name], kde=True, bins=30)
    plt.title("Histogram of " + feature_name)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    file_path = os.path.join(FEATURE_DATA_FOLDER, f"{feature_name}_histogram.png")
    plt.savefig(file_path)
    plt.close()


def boxplot(df, feature_name):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Label", y=feature_name, data=df)
    plt.title("Boxplot " + feature_name)
    file_path = os.path.join(FEATURE_DATA_FOLDER, f"{feature_name}_boxplot.png")
    plt.savefig(file_path)
    plt.close()


def visualise_features():
    x, y = get_all_features("segmented_data")
    df = features_to_dataframe(x, y)
    for feature in FEATURES:
        boxplot(df, feature)
        histogram(df, feature)


visualise_features()
