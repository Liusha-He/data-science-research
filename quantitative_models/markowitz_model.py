from typing import Tuple, Callable, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimization


def plot_sharpe_ratio(
    returns: np.ndarray, risks: np.ndarray
):
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, returns,
                c=get_sharpe_ratio(returns, risks),
                marker="o")
    plt.grid(True)
    plt.xlabel("Expected Risks")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


def calculate_portfolio_return(
    weights: np.ndarray, returns: pd.DataFrame, n_of_trading_days: int
) -> float:
    portfolio_return = np.sum(returns.mean() * weights) * n_of_trading_days
    return portfolio_return


def calculate_portfolio_risk(
    weights: np.ndarray, returns: pd.DataFrame, n_of_trading_days: int
) -> float:
    portfolio_risk = np.sqrt(
        np.dot(
            weights.T, np.dot(returns.cov() * n_of_trading_days, weights)
        )
    )

    return portfolio_risk


def get_sharpe_ratio(
    returns: Union[np.ndarray, float], risks: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    return returns / risks


def min_function_sharpe(
    weights: np.ndarray, returns: pd.DataFrame, n_of_trading_days: int
) -> float:
    portfolio_return = calculate_portfolio_return(weights, returns, n_of_trading_days)
    portfolio_risk = calculate_portfolio_risk(weights, returns, n_of_trading_days)

    return -get_sharpe_ratio(portfolio_return, portfolio_risk)


class SharpeRatioOptimizer:
    def __init__(self,
                 weights: List,
                 returns: pd.DataFrame,
                 n_of_asset: int,
                 func: Callable,
                 n_of_trading_days: int,
                 method: str = "SLSQP"):
        self.weights = weights
        self.returns = returns
        self.target_func = func
        self.n_of_trading_days = n_of_trading_days
        self.method = method
        self.constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1
        }
        self.bounds = tuple(
            (0, 1) for _ in range(n_of_asset)
        )

        optimal_weights = self._optimize()["x"].round(3)
        optimal_ratio = -self.target_func(self.weights, self.returns, self.n_of_trading_days)

        self.result = {
            "weights": {
                k: v for k, v in zip(returns.columns, optimal_weights)
            },
            "sharpe_ratio": round(optimal_ratio, 3),
            "expected_return": round(calculate_portfolio_return(
                optimal_weights, returns, n_of_trading_days
            ), 3),
            "expected_risk": round(calculate_portfolio_risk(
                optimal_weights, returns, n_of_trading_days
            ), 3)
        }

    def _optimize(self):
        return optimization.minimize(
            fun=self.target_func,
            x0=self.weights[0],
            args=(
                self.returns,
                self.n_of_trading_days
            ),
            method=self.method,
            bounds=self.bounds,
            constraints=self.constraints,
        )


class MarkowitzModel:
    def __init__(self,
                 history_data: pd.DataFrame,
                 n_of_simulations: int = 10_000,
                 log_return: bool = True,
                 length_of_trading: int = 252):
        self.df = history_data

        self.n_of_simulation = n_of_simulations
        self._log_return = log_return
        self.n_of_trading_days = length_of_trading

    @property
    def covariance(self) -> pd.DataFrame:
        return self.stock_returns.cov() * self.n_of_trading_days

    @property
    def stock_returns(self) -> pd.DataFrame:
        if self._log_return:
            return np.log(self.df / self.df.shift(1))[1:]
        return self.df.pct_change()[1:]

    @property
    def stock_periodical_returns(self) -> float:
        return self.stock_returns.mean() * self.n_of_trading_days

    @property
    def weights(self) -> List[np.ndarray]:
        n_of_assets = self.df.shape[1]

        weights = []
        for _ in range(self.n_of_simulation):
            w = np.random.random(n_of_assets)
            w /= np.sum(w)
            weights.append(w)

        return weights

    def generate_portfolios(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        portfolio_means = []
        portfolio_risks = []

        portfolio_weights = self.weights

        for w in portfolio_weights:
            portfolio_means.append(
                calculate_portfolio_return(w, self.stock_returns, self.n_of_trading_days)
            )
            portfolio_risks.append(
                calculate_portfolio_risk(w, self.stock_returns, self.n_of_trading_days)
            )

        portfolio_means = np.array(portfolio_means)
        portfolio_risks = np.array(portfolio_risks)

        portfolio_weights = np.array(portfolio_weights)

        plot_sharpe_ratio(portfolio_means, portfolio_risks)

        return (
            portfolio_weights, portfolio_means, portfolio_risks
        )

    def sharpe_ratio_optimize(self) -> SharpeRatioOptimizer:
        n_of_assets = self.df.shape[1]
        weights = np.random.random(n_of_assets)
        weights /= np.sum(weights)
        return SharpeRatioOptimizer(
            weights=weights,
            returns=self.stock_returns,
            n_of_asset=self.df.shape[1],
            func=min_function_sharpe,
            n_of_trading_days=self.n_of_trading_days,
        )
