import pandas.api.types as ptypes
import pytest

from iac_exercise import features
from iac_exercise.config.core import config


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


# Testing FakeVariableTransformer
def test_type_fake_var_transformer(sample_input_data):
    transformer = features.FakeVariableTransformer(*config.model_config.fake_create_var)

    subject = transformer.fit_transform(sample_input_data)

    assert ptypes.is_float_dtype(subject[config.model_config.fake_var])


def test_shape_fake_var_transformer(sample_input_data):
    transformer = features.FakeVariableTransformer(*config.model_config.fake_create_var)

    expected_df_shape = (49, 7)
    subject = transformer.fit_transform(sample_input_data)

    assert subject.shape == expected_df_shape


def test_value_fake_var_transformer(sample_input_data):
    transformer = features.FakeVariableTransformer(*config.model_config.fake_create_var)

    expected_value = 0.078
    subject = transformer.fit_transform(sample_input_data)

    assert subject[config.model_config.fake_var][0] == pytest.approx(
        expected_value, 0.1
    )
