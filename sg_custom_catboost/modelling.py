import joblib
import pandas as pd
from typing import Dict
from sg_custom_catboost import utils, data_manager
from sg_custom_catboost.pipeline import drinks_pipeline
from sg_custom_catboost.config.core import config
from catboost import CatBoostRegressor


def training_model(model_data_dict: Dict) -> Dict:
    """

    Parameters
    ----------
    model_data_dict

    Returns
    -------

    """
    data_features = model_data_dict[config.app_config.model_data_features]
    data_target = model_data_dict[config.app_config.model_data_target]

    model = CatBoostRegressor(
        iterations=config.model_config.catboost_itr,
        learning_rate=config.model_config.catboost_lr,
        logging_level=config.model_config.catboost_logging_state
    )

    model.fit(
        data_features,
        data_target
    )

    predictions = model.predict(
        data_features
    )

    model_data_dict[config.app_config.model_data_model] = model
    model_data_dict[config.app_config.model_data_predictions] = predictions

    return model_data_dict


def evaluating_model(model_data_dict: Dict) -> Dict:
    """

    Parameters
    ----------
    model_data_dict

    Returns
    -------

    """

    data_features = model_data_dict[config.app_config.model_data_features]
    data_target = model_data_dict[config.app_config.model_data_target]

    model = data_manager.load_model(config.model_config.model_save_file)

    predictions = model.predict(data_features)

    model_data_dict[config.app_config.model_data_predictions] = predictions

    return model_data_dict
