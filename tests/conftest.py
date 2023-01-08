import pytest
import pandas as pd


@pytest.fixture
def stocks_data():
    return pd.read_csv("data/stocks.csv")
