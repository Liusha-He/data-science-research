import os
import pytest
import pandas as pd


@pytest.fixture
def stocks_data():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data/stocks.csv"),
        index_col=["Date"],
        parse_dates=["Date"]
    )

    return df.loc[:"2022-01-01"]
