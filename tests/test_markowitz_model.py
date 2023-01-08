import matplotlib.pyplot as plt
from quantitative_models.markowitz_model import MarkowitzModel


def test_markowitz_model_by_example(stocks_data):
    model = MarkowitzModel(history_data=stocks_data, length_of_trading=150)

    assert len(model.weights) == 10_000
    assert model.sharpe_ratio_optimize().result["weights"]["DB"] > 0.7
