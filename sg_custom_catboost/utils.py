import json
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sg_custom_catboost.config.core import METRICS_DIR, config


def fix_standard_scaler_variables(
    variables_list: List,
    target_dataframe: pd.DataFrame,
    original_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return proper variables without Standard Scaler
    transformation.

    Args:
        variables_list (List): List of variables to avoid transformation.
        target_dataframe (pd.DataFrame): The transformed dataframe.
        original_dataframe (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: The fixed_dataframe.
    """

    fixed_dataframe = target_dataframe.copy()
    for var in variables_list:
        fixed_dataframe[var] = original_dataframe[var]

    return fixed_dataframe


def standard_scaler_dataframe(
    target_dataframe: pd.DataFrame, original_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Applies StandardScaler method and return a structured dataframe.

    Args:
        target_dataframe (pd.DataFrame): The transformed dataframe.
        original_dataframe (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: [description]
    """

    transformed_dataframe = StandardScaler().fit_transform(target_dataframe)
    transformed_dataframe = pd.DataFrame(
        transformed_dataframe,
        index=original_dataframe.index,
        columns=original_dataframe.columns,
    )

    return transformed_dataframe


def drop_target_variable(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the target variable from test dataset.

    Parameters
    ----------
    dataframe: Processed dataframe for model estimation.

    Returns
    -------
    Dataframe without the target variable.

    """

    df = dataframe.copy()
    df = df.drop(config.model_config.target, axis=1)

    return df


def generate_regression_metrics(
    y_test: pd.DataFrame, y_pred: pd.DataFrame, verbose=False
):
    """

    Parameters
    ----------
    y_test
    y_pred

    Returns
    -------

    """
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)

    metrics_dict = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    if verbose:
        print("Mean Absolute Error    :", round(mae, 3))
        print("Mean Squared Error     :", round(mse, 3))
        print("Root Mean Squared Error:", round(rmse, 3))
        print("R2:", round(r2, 3))

    return metrics_dict


def save_regression_metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    metrics_dict = generate_regression_metrics(y_test, y_pred, verbose=True)

    timestamp = define_timestamp()

    metrics_timestamp = "{}_{}.json".format(
        config.app_config.metrics_file_name, timestamp
    )

    train_data_name_latest = "{}_{}.json".format(
        config.app_config.metrics_file_name, config.app_config.latest_timestamp
    )

    metrics_path_timestamp = os.path.join(METRICS_DIR, metrics_timestamp)

    metrics_path_latest = os.path.join(METRICS_DIR, train_data_name_latest)

    metrics_path_list = [metrics_path_timestamp, metrics_path_latest]

    for path in metrics_path_list:
        with open(path, "w") as fp:
            json.dump(metrics_dict, fp)


def define_timestamp():
    """
    Return the actual timeframe.

    Returns
    -------
    A string with the actual timeframe.

    """
    timeframe = str(pd.to_datetime("today"))[0:19]
    return timeframe


def filter_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe of wrong values.
    Parameters
    ----------
    df : The original dataframe.
    Returns
    -------
    The dataframe with its data properly filtered.
    """
    df = dataframe.copy()
    df = df[~df[config.model_config.filter_ph].isnull()]
    df = df[df[config.model_config.filter_ph] <= config.model_config.ph_limit]
    df = df[df[config.model_config.filter_ebc] <= config.model_config.ebc_limit]
    df = df[df[config.model_config.filter_ibu] <= config.model_config.ibu_limit]
    df = df[df[config.model_config.filter_srm] <= config.model_config.srm_limit]
    return df


def create_data_split(dataframe: pd.DataFrame):
    """

    Parameters
    ----------
    dataframe

    Returns
    -------

    """
    data_features = dataframe.drop(config.model_config.target, axis=1)

    data_target = dataframe[config.model_config.target]

    model_data_dict = {
        config.app_config.model_data_features: data_features,
        config.app_config.model_data_target: data_target,
    }

    return model_data_dict


def create_processed_data_split(dataframe: pd.DataFrame, pipeline):
    """

    Parameters
    ----------
    dataframe
    pipeline

    Returns
    -------

    """
    model_data = pipeline.fit_transform(dataframe)

    processed_model_data_dict = create_data_split(model_data)

    return processed_model_data_dict
