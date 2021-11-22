from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from sg_custom_catboost import features
from sg_custom_catboost.config.core import config


transformation_pipeline = [
    (
        config.model_config.target_diff_transformer_name, 
        features.TargetDiffTransformer(
            *config.model_config.diff_create_var
            )
        ),
    (
        config.model_config.drop_variables_transformer_name, 
        features.SelectVariablesTransformers(
            config.model_config.to_drop_train
            )
        ),
    (
        config.model_config.standard_scaler_transformer_name, 
        features.StandardScalerTransformers(
            config.model_config.no_standard_scaler_variables
            )
        )
]


drinks_pipeline = Pipeline(transformation_pipeline)
