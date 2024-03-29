import catboost
import io
import requests
import pandas as pd
from scipy.sparse import hstack
from zipfile import ZipFile
import gzip
import idx2numpy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from presc.dataset import Dataset
from presc.copies.sampling import dynamical_range, mixed_data_features


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
        dataset["recommend"] = dataset["recommend"].astype("category")
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
        dataset["Occupancy"] = dataset["Occupancy"].astype("category")
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
        dataset.drop("region_pixel_count", axis=1, inplace=True)
        dataset["class"] = dataset["class"].astype("category")
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


class AutismScreeningModel:
    """Classifier model for the Autistic Spectrum Adult Screening dataset."""

    def __init__(self):
        # Obtain and preprocess dataset
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00426/"
            "Autism-Adult-Data%20Plus%20Description%20File.zip"
        )
        compressed_file = ZipFile(io.BytesIO(requests.get(url).content))
        feature_names = [
            "A1_Score",
            "A2_Score",
            "A3_Score",
            "A4_Score",
            "A5_Score",
            "A6_Score",
            "A7_Score",
            "A8_Score",
            "A9_Score",
            "A10_Score",
            "age",
            "gender",
            "ethnicity",
            "jundice",
            "autism",
            "country_of_res",
            "used_app_before",
            "result",
            "age_desc",
            "relation",
            "ASD",
        ]
        dtype = {
            "age": "Int64",
            "gender": "category",
            "jundice": "category",
            "autism": "category",
            "A1_Score": "category",
            "A2_Score": "category",
            "A3_Score": "category",
            "A4_Score": "category",
            "A5_Score": "category",
            "A6_Score": "category",
            "A7_Score": "category",
            "A8_Score": "category",
            "A9_Score": "category",
            "A10_Score": "category",
            "ASD": "category",
        }
        dataset = pd.read_csv(
            compressed_file.open("Autism-Adult-Data.arff"),
            skiprows=25,
            names=feature_names,
            na_values="?",
            dtype=dtype,
        )
        dataset = dataset.drop(
            columns=[
                "ethnicity",
                "country_of_res",
                "used_app_before",
                "result",
                "age_desc",
                "relation",
            ]
        )
        dataset = dataset[
            ["age", "gender", "jundice", "autism"]
            + dataset.columns[:10].to_list()
            + ["ASD"]
        ]
        dataset = dataset.dropna()
        dataset["age"] = dataset["age"].astype("int64")
        dataset["ASD"] = dataset["ASD"].cat.rename_categories(
            new_categories={"NO": 0, "YES": 1}
        )
        self.dataset = Dataset(dataset, label_col="ASD")

        # Obtain feature space description
        self.feature_description = mixed_data_features(self.dataset.features)
        self.feature_description["age"]["max"] = 100

        # Split dataset in test and train samples
        X = self.dataset.features
        y = self.dataset.labels
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(
            X, y, test_size=0.20, random_state=0, stratify=y
        )

        categorical_features = [
            "gender",
            "jundice",
            "autism",
            "A1_Score",
            "A2_Score",
            "A3_Score",
            "A4_Score",
            "A5_Score",
            "A6_Score",
            "A7_Score",
            "A8_Score",
            "A9_Score",
            "A10_Score",
        ]
        train_pool = catboost.Pool(
            self.X_train, self.y_train, cat_features=categorical_features
        )

        # Gradient Boosting on Decision Trees Model
        self.model = catboost.CatBoostClassifier()
        self.model.fit(train_pool, verbose=False)


class QuestionPairsModel:
    """Classifier model for the Quora Question Pairs Dataset."""

    def __init__(self, **fit_parameters):
        url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
        dataset = pd.read_csv(url, sep="\t")
        dataset = dataset[["question1", "question2", "is_duplicate"]].dropna()
        self.dataset = Dataset(dataset, label_col="is_duplicate")

        X = self.dataset.features
        y = self.dataset.labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )

        # Instantiate transformer and classifier
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=0.000002)
        self.classifier = LogisticRegression(solver="liblinear", random_state=0)

        # Fit vectorizer
        all_questions = pd.concat([X["question1"], X["question2"]])
        self.count_vectorizer.fit(all_questions)

        # Tranform training data
        X_train_q1 = self.count_vectorizer.transform(X["question1"])
        X_train_q2 = self.count_vectorizer.transform(X["question2"])
        X_train_q1q2 = hstack([X_train_q1, X_train_q2])

        # Fit classifier
        self.classifier.fit(X_train_q1q2, y, **fit_parameters)

    def predict(self, X):
        # Transform query data
        X_predict_q1 = self.count_vectorizer.transform(X.iloc[:, 0])
        X_predict_q2 = self.count_vectorizer.transform(X.iloc[:, 1])
        X_predict_q1q2 = hstack([X_predict_q1, X_predict_q2])

        # Predict with classifier
        y_predict = self.classifier.predict(X_predict_q1q2)
        return y_predict


class CustomFlattener(BaseEstimator, TransformerMixin):
    """Convert 2D numpy arrays into 1D pandas columns."""

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        X_transformed = X.apply(
            lambda x: pd.Series(x["images"].reshape(1, -1)[0]), axis=1
        )
        return X_transformed


class DigitsModel:
    """Classifier model for the MNIST Handwritten Digits Dataset."""

    def __init__(self):
        # Obtain and preprocess dataset
        url_prefix = "http://yann.lecun.com/exdb/mnist/"
        file_names = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]
        data_idx_all = {}
        for name in file_names:
            data_idx_all[name[:-3]] = gzip.decompress(
                requests.get(url_prefix + name).content
            )

        X_train_array = idx2numpy.convert_from_string(
            data_idx_all["train-images-idx3-ubyte"]
        )
        self.X_train = pd.DataFrame(
            {"images": [image for image in X_train_array.astype(int)]}
        )
        self.y_train = pd.DataFrame(
            (
                idx2numpy.convert_from_string(data_idx_all["train-labels-idx1-ubyte"])
            ).astype(int),
            columns=["labels"],
            dtype="category",
        )

        X_test_array = idx2numpy.convert_from_string(
            data_idx_all["t10k-images-idx3-ubyte"]
        )
        self.X_test = pd.DataFrame(
            {"images": [image for image in X_test_array.astype(int)]}
        )
        self.y_test = pd.DataFrame(
            (
                idx2numpy.convert_from_string(data_idx_all["t10k-labels-idx1-ubyte"])
            ).astype(int),
            columns=["labels"],
            dtype="category",
        )

        self.dataset = Dataset(
            pd.DataFrame(
                pd.concat(
                    [
                        pd.concat([self.X_train, self.X_test], ignore_index=True),
                        pd.concat([self.y_train, self.y_test], ignore_index=True),
                    ],
                    axis=1,
                )
            ),
            label_col="labels",
        )

        # Image flattener from 2D to 1D and SVC Model
        self.model = Pipeline(
            [
                ("flattener", CustomFlattener()),
                ("SVC_classifier", SVC(kernel="rbf")),
            ]
        )
        self.model = self.model.fit(self.X_train, y=self.y_train.iloc[:, 0])
