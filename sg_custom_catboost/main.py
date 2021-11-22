import pandas as pd
import pickle
import os
from sg_custom_catboost import pipeline, data_manager
from sg_custom_catboost.config.core import PROCESSED_DATASET_DIR, config
from sg_custom_catboost.features import *

def features() -> None:
    """
    Generates the features to create the train and test dataframes for
    model stage.

    Parameters
    ----------
    df_path (str): Train data path.
    """

    timeframe = str(pd.to_datetime("today"))[0:19]

    train_data_name_timeframe = "{}_{}.csv".format(
        config.app_config.processed_train_data,
        timeframe
    )

    train_data_name_latest = "{}_{}.csv".format(
        config.app_config.processed_train_data,
        config.app_config.latest_timestamp
    )

    train_data_path_timestamp = os.path.join(
        PROCESSED_DATASET_DIR,
        train_data_name_timeframe
    )

    train_data_path_latest = os.path.join(
        PROCESSED_DATASET_DIR,
        train_data_name_latest
    )

    train_data = data_manager.load_dataset(
        file_name=config.app_config.training_data_file
    )

    pipeline = data_manager.load_pipeline(
        file_name=config.app_config.pipeline_name
    )

    model_data = pipeline.fit_transform(train_data)

    model_data.to_csv(
        train_data_path_timestamp,
        index=False
    )


    model_data.to_csv(
        train_data_path_latest,
        index=False
    )



