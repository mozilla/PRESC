import io
import requests
import pandas as pd
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from presc.dataset import Dataset
from presc.copies.sampling import dynamical_range


class WinesModel:
    """Classifier model for the wine quality dataset."""

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

        self.scaler = StandardScaler().fit(self.X_train)

        X_train_scaled = self.scaler.transform(self.X_train)

        # KNN Model
        self.model = KNeighborsClassifier(n_neighbors=30, weights="distance")
        self.model = self.model.fit(X_train_scaled, self.y_train)


class OccupancyModel:
    """Classifier model for the room occupancy dataset."""

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

        self.scaler = StandardScaler().fit(self.X_train)
        X_train_scaled = self.scaler.transform(self.X_train)

        # Decision Tree Model
        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=15)
        self.model = self.model.fit(X_train_scaled, self.y_train)


class SegmentationModel:
    """Classifier model for the image segmentation dataset."""

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

        self.scaler = StandardScaler().fit(self.X_train)
        X_train_scaled = self.scaler.transform(self.X_train)

        # SVC Model
        self.model = SVC(
            kernel="rbf", decision_function_shape="ovr", class_weight="balanced", C=5
        )
        self.model.fit(X_train_scaled, self.y_train)
