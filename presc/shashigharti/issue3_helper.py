from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import warnings


def fix_outlier_with_boundary_value(df, df_columns_with_outliers):
    """ This function takes data as input and replaces outliers > max value of each columns by max value
    
    Args:
        df: dataframe of all data
    Returns:
        data_df: processed dataframe
        
    """
    data_df = df.copy()

    # Fill null
    data_df.fillna(data_df.mean(), inplace=True)

    # Replace outliers with max boundary value
    for column in enumerate(df_columns_with_outliers.columns):
        data_df.loc[
            data_df[column] > df_columns_with_outliers[column][0], column
        ] = df_columns_with_outliers[column][0]

    data_df["Class"] = pd.Categorical(data_df["Class"]).codes

    return data_df


def test_train_split(estimator, X, y, scaler=None):
    """ This function takes estimator and data as input and does test-train split test in multiple passes.

    Args:
        estimator: Model to be trained
        X: Training Data(features columns)
        y: Labels for each row

    Returns:
        dataframe: list of test-train split results
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)

    result_lst = list()

    for i in range(10, 100, 5):

        # Calculate Test/Train Ratio
        test_ratio = round(i / 100, 2)
        train_ratio = round(1 - test_ratio, 2)

        # Split the data based on the calculated test ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=1
        )

        if scaler is not None:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        result_lst.append(
            [
                train_ratio,
                test_ratio,
                metrics.accuracy_score(y_test, y_pred),
                metrics.f1_score(y_test, y_pred, average="weighted"),
            ]
        )

        # Convert the list to dataframe
        result_df = pd.DataFrame(
            result_lst,
            columns=["Ratio(Train Data)", "Ratio(Test Data)", "Accuracy", "F1-Score"],
        )

    print(result_df)

    return result_df
