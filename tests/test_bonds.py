from quantitative_models.bonds import ZeroCouponBond, CouponBond


def test_zerocouponbond_by_example():
    principal = 1_000
    interest_rate = 5
    maturity = 3

    bond = ZeroCouponBond(
        principal=principal, maturity=maturity, interest_rate=interest_rate
    )

    assert bond.price == 863.84


def test_couponbond_by_example():
    principal = 1_000
    coupon_rate = 10
    maturity = 5
    interest_rate = 4

    bond = CouponBond(
        principal=principal, coupon_rate=coupon_rate, maturity=maturity, interest_rate=interest_rate
    )

    assert bond.price == 1267.11
