import io
import requests
import pandas as pd
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from presc.dataset import Dataset
from presc.copies.sampling import dynamical_range


class WinesModel:
    """Classifier model for the Wine Quality dataset."""

    def __init__(self):
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "wine-quality/winequality-white.csv"
        )
        dataset = pd.read_csv(url, sep=";")
        dataset["recommend"] = 0
        dataset.loc[dataset["quality"] > 6, "recommend"] = 1
        dataset = dataset.drop("quality", axis=1)
        self.dataset = Dataset(dataset, label_col="recommend")

        self.feature_description = dynamical_range(self.dataset.features, verbose=False)

        X = self.dataset.features
        y = self.dataset.labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.6, random_state=0, stratify=y
        )

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "KKN_classifier",
                    KNeighborsClassifier(n_neighbors=30, weights="distance"),
                ),
            ]
        )

        # KNN Model
        self.model = self.model.fit(self.X_train, self.y_train)


class OccupancyModel:
    """Classifier model for the Room Occupancy Detection dataset."""

    def __init__(self):
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00357/occupancy_data.zip"
        )
        compressed_file = ZipFile(io.BytesIO(requests.get(url).content))
        dataset = pd.read_csv(compressed_file.open("datatraining.txt"))
        dataset = dataset.loc[:, dataset.columns != "date"]
        self.dataset = Dataset(dataset, label_col="Occupancy")

        X = self.dataset.features
        y = self.dataset.labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.6, random_state=0, stratify=y
        )

        self.feature_description = dynamical_range(self.dataset.features, verbose=False)

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "tree_classifier",
                    DecisionTreeClassifier(max_depth=5, min_samples_split=15),
                ),
            ]
        )

        # Decision Tree Model
        self.model = self.model.fit(self.X_train, self.y_train)


class SegmentationModel:
    """Classifier model for the Image Segmentation dataset."""

    def __init__(self):
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "image/segmentation.test"
        )
        feature_names = [
            "class",
            "region_centroid_col",
            "region_centroid_row",
            "region_pixel_count",
            "short_line_density_5",
            "short_line_density_2",
            "vedge_mean",
            "vegde_sd",
            "hedge_mean",
            "hedge_sd",
            "intensity_mean",
            "rawred_mean",
            "rawblue_mean",
            "rawgreen_mean",
            "exred_mean",
            "exblue_mean",
            "exgreen_mean",
            "value_mean",
            "saturation_mean",
            "hue_mean",
        ]
        dataset = pd.read_csv(
            url, skiprows=4, names=feature_names, dtype={"class": str}
        )
        dataset = dataset[feature_names[1:] + ["class"]]
        self.dataset = Dataset(dataset, label_col="class")

        self.feature_description = dynamical_range(self.dataset.features, verbose=False)

        X = self.dataset.features
        y = self.dataset.labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.4, random_state=0, stratify=y
        )

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "SVC_classifier",
                    SVC(
                        kernel="rbf",
                        decision_function_shape="ovr",
                        class_weight="balanced",
                        C=5,
                    ),
                ),
            ]
        )

        # SVC Model
        self.model = self.model.fit(self.X_train, self.y_train)
