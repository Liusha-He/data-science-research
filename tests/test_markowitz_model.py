import matplotlib.pyplot as plt
from quantitative_models.markowitz_model import MarkowitzModel


def test_markowitz_model_by_example(stocks_data):
    model = MarkowitzModel(history_data=stocks_data, length_of_trading=252)

    assert len(model.weights) == 10_000

    result = model.sharpe_ratio_optimize().result

    assert result["expected_return"] > 0
    assert result["expected_risk"] < 0.5
    assert result["sharpe_ratio"] > 0.5

    assert result["weights"]["AAPL"] > 0
    assert result["weights"]["MSFT"] > 0

    assert result["weights"]["DB"] == 0
