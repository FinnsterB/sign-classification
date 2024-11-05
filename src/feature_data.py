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
    "Area",
]

FEATURE_DATA_FOLDER = "feature_plots"


def features_to_dataframe(x, y):
    df = pd.DataFrame(
        x,
        columns=FEATURES,
    )
    df["Label"] = y
    return df


def histogram(df, feature_name):
    histogram_folder = FEATURE_DATA_FOLDER + "/histograms"
    os.makedirs(histogram_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature_name], kde=True, bins=30)
    plt.title("Histogram of " + feature_name)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    file_path = os.path.join(histogram_folder, f"{feature_name}_histogram.png")
    plt.savefig(file_path)
    plt.close()


def boxplot(df, feature_name):
    boxplot_folder = FEATURE_DATA_FOLDER + "/boxplots"
    os.makedirs(boxplot_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Label", y=feature_name, data=df)
    plt.title("Boxplot " + feature_name)
    file_path = os.path.join(boxplot_folder, f"{feature_name}_boxplot.png")
    plt.savefig(file_path)
    plt.close()


def scatterplot(df, feature1, feature2):
    NR_OF_SIGNS = 5
    scatterplot_folder = FEATURE_DATA_FOLDER + "/scatterplots"
    os.makedirs(scatterplot_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=feature1,
        y=feature2,
        hue="Label",
        data=df,
        palette=sns.color_palette("husl", NR_OF_SIGNS),
    )
    plt.title(f"Scatter plot of {feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    file_path = os.path.join(
        scatterplot_folder, f"scatterplot_{feature1}_vs_{feature2}.png"
    )
    plt.savefig(file_path)
    plt.close()


def correlation_matrix(df):
    correlation_folder = os.path.join(FEATURE_DATA_FOLDER, "correlation_matrices")
    os.makedirs(correlation_folder, exist_ok=True)
    corr = df[FEATURES].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar=True
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    file_path = os.path.join(correlation_folder, "correlation_matrix.png")
    plt.savefig(file_path)
    plt.close()


def generate_feature_plots():
    x, y = get_all_features("segmented_data")
    df = features_to_dataframe(x, y)
    for feature in FEATURES:
        boxplot(df, feature)
        histogram(df, feature)

    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            scatterplot(df, FEATURES[i], FEATURES[j])

    correlation_matrix(df)


generate_feature_plots()
