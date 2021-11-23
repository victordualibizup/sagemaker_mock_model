import numpy as np
import pytest

from sg_custom_catboost.predict import make_prediction


def test_make_prediction(sample_input_data):
    expected_no_predictions = 40
    expected_first_prediction_value = 49

    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert len(predictions) == expected_no_predictions
    assert predictions[0] == pytest.approx(expected_first_prediction_value, 0.1)
