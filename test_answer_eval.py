import unittest
import pytest
from answer_eval import get_eval_model


@pytest.mark.parametrize("model_name", ['gpt-4', 'gpt-3.5-turbo'])
def test_get_eval_model(model_name):
    model = get_eval_model(model_name)
    assert model.model_name == model_name
    assert model.temperature == 0


def test_get_eval_model_with_invalid_model_name():
    with pytest.raises(ValueError):
        get_eval_model("invalid_model")


if __name__ == '__main__':
    unittest.main()
