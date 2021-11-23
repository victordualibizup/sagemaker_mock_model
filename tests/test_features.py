import pandas.api.types as ptypes
import pytest

from sg_custom_catboost import features
from sg_custom_catboost.config.core import config


# Testing TargetDiffTransformer
def test_type_target_diff_transformer(sample_input_data):
    transformer = features.TargetDiffTransformer(*config.model_config.diff_create_var)

    subject = transformer.fit_transform(sample_input_data)

    assert ptypes.is_float_dtype(subject[config.model_config.target_dg])


def test_shape_target_diff_transformer(sample_input_data):
    transformer = features.TargetDiffTransformer(*config.model_config.diff_create_var)

    expected_df_shape = (49, 7)
    subject = transformer.fit_transform(sample_input_data)

    assert subject.shape == expected_df_shape


def test_value_target_diff_transformer(sample_input_data):
    transformer = features.TargetDiffTransformer(*config.model_config.diff_create_var)

    expected_value = -42.0
    subject = transformer.fit_transform(sample_input_data)

    assert subject[config.model_config.target_dg][0] == expected_value
