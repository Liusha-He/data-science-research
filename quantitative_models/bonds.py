from abc import ABC
from typing import Optional
import numpy as np


class BondInterface(ABC):
    principal: float
    interest_rate: float
    maturity: int
    coupon_rate: Optional[float]

    def __init__(self,
                 principal: float,
                 maturity: int,
                 interest_rate: float,
                 coupon_rate: float = None,):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate / 100
        self.coupon_rate = coupon_rate / 100 if coupon_rate is not None else None

    def present_value(self, x: float, n: int) -> float:
        return round(x * np.exp(-self.interest_rate * n), 2)

    @property
    def price(self) -> float:
        return self.present_value(self.principal, self.maturity)


class ZeroCouponBond(BondInterface):
    def __init__(self, principal: float, maturity: int, interest_rate: float):
        super().__init__(principal, maturity, interest_rate)

    def present_value(self, x: float, n: int) -> float:
        return round(x / (1 + self.interest_rate) ** n, 2)


class CouponBond(BondInterface):
    def __init__(self, principal: float, coupon_rate: float, maturity: int, interest_rate: float):
        super().__init__(principal, maturity, interest_rate, coupon_rate)

    def present_value(self, x: float, n: int) -> float:
        return round(x / (1 + self.interest_rate) ** n, 2)
    
    @property
    def price(self) -> float:
        price = 0

        for t in range(1, self.maturity + 1):
            price += self.present_value(self.principal*self.coupon_rate, t)

        return price + self.present_value(self.principal, self.maturity)
