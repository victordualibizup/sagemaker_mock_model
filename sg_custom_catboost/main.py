import pandas as pd
import pickle
import os
import fire
from sklearn import metrics
from sg_custom_catboost import pipeline, data_manager, utils
from sg_custom_catboost.config.core import PROCESSED_DATASET_DIR, config
from catboost import CatBoostRegressor


# TODO: REFACTOR THIS FUNCTION
def features() -> None:
    """
    Generates the features to create the train and test dataframes for
    model stage.

    Parameters
    ----------
    df_path (str): Train data path.
    """

    timestamp = utils.define_timestamp()

    train_data_name_timestamp = "{}_{}.csv".format(
        config.app_config.processed_train_data,
        timestamp
    )

    train_data_name_latest = "{}_{}.csv".format(
        config.app_config.processed_train_data,
        config.app_config.latest_timestamp
    )

    train_data_path_timestamp = os.path.join(
        PROCESSED_DATASET_DIR,
        train_data_name_timestamp
    )

    train_data_path_latest = os.path.join(
        PROCESSED_DATASET_DIR,
        train_data_name_latest
    )

    test_data_name_timestamp = "{}_{}.csv".format(
        config.app_config.processed_test_data,
        timestamp
    )

    test_data_name_latest = "{}_{}.csv".format(
        config.app_config.processed_test_data,
        config.app_config.latest_timestamp
    )

    test_data_path_timestamp = os.path.join(
        PROCESSED_DATASET_DIR,
        test_data_name_timestamp
    )

    test_data_path_latest = os.path.join(
        PROCESSED_DATASET_DIR,
        test_data_name_latest
    )

    train_data = data_manager.load_dataset(
        file_name=config.app_config.training_data_file
    )

    test_data = data_manager.load_dataset(
        file_name=config.app_config.new_data_file
    )

    # TODO: CREATE AND SAVE PIPELINE
    pipeline = data_manager.load_pipeline(
        file_name=config.app_config.pipeline_name
    )

    train_model_data = pipeline.fit_transform(train_data)
    test_model_data = pipeline.fit_transform(test_data)

    train_model_data.to_csv(
        train_data_path_timestamp,
        index=False
    )

    train_model_data.to_csv(
        train_data_path_latest,
        index=False
    )

    test_model_data.to_csv(
        test_data_path_timestamp,
        index=False
    )

    test_model_data.to_csv(
        test_data_path_latest,
        index=False
    )


# TODO: REFACTOR THIS FUNCTION
def train_model() -> None:
    """

    Returns
    -------

    """
    processed_train_data = data_manager.load_dataset(
        file_name=config.app_config.latest_train_data,
        raw_data=False
    )

    processed_test_data = data_manager.load_dataset(
        file_name=config.app_config.latest_test_data,
        raw_data=False
    )

    processed_train_data = utils.filter_dataframe(
        processed_train_data
    )

    processed_test_data = utils.filter_dataframe(
        processed_test_data
    )

    X_train = processed_train_data.drop(
        config.model_config.target,
        axis=1
    )
    X_test = processed_test_data.drop(
        config.model_config.target,
        axis=1
    )

    y_train = processed_train_data[config.model_config.target]
    y_test = processed_test_data[config.model_config.target]

    model = CatBoostRegressor(
        iterations=config.model_config.catboost_itr,
        learning_rate=config.model_config.catboost_lr,
        logging_level=config.model_config.catboost_logging_state
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )
    # TODO: CREATE MODEL TIMESTAMP AND LATEST
    utils.save_regression_metrics(y_test, y_pred)
    data_manager.save_model(
        config.model_config.model_save_file
    )


def run():
    """
    Runs all model pipeline sequentially.
    Returns
    -------

    """
    features()
    train_model()


def cli():
    """ Caller to transform module in a low-level CLI """
    return fire.Fire()


if __name__ == "__main__":
    from sg_custom_catboost.features import *

    cli()
