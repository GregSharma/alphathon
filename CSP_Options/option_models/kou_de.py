import logging

import numpy as np
from fypy.fit.Calibrator import Calibrator, LeastSquares
from fypy.fit.Loss import Loss
from fypy.fit.Minimizer import ScipyMinimizer
from fypy.fit.Targets import Targets
from fypy.model.levy import KouJD
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv

import csp
from csp import ts
from CSP_Options.constants import RISK_FREE_RATE
from CSP_Options.structs import VectorizedOptionQuote, VolFit


class RMSE(Loss):
    """sqrt( sum of squared residuals / N )"""

    def residual_apply(self, x):
        return np.power(x, 2)

    def agg_apply(self, s):
        return np.sqrt(s[0] / len(s))


@csp.node
def node_kde_fit(vq: ts[VectorizedOptionQuote], spot_price: ts[float]) -> ts[VolFit]:
    with csp.state():
        initial_params = {"sigma": 0.14, "lam": 2.0, "p_up": 0.3, "eta1": 20, "eta2": 15}
        s_initial_guess = np.array(list(initial_params.values()))
        disc_curve = DiscountCurve_ConstRate(rate=RISK_FREE_RATE)
        fwd = EquityForward(
            S0=12345678, discount=disc_curve
        )  # 12345678 is a placeholder. When we handle inputs below, we will replace it with the true spot_price.
        model = KouJD(
            **initial_params,
            forwardCurve=fwd,
            discountCurve=DiscountCurve_ConstRate(rate=RISK_FREE_RATE),
        )
        pricer = ProjEuropeanPricer(model=model, N=2**11, L=16)
        calibrator = Calibrator(
            model=model,
            minimizer=LeastSquares(ftol=0.0316227766),  # , max_nfev=5),
            # minimizer=ScipyMinimizer(method="L-BFGS-B"),
            loss=RMSE(),
        )
        s_fitted_iv = None

    if csp.ticked(vq):
        strike, prices, ttm, flags, iv = vq.strike, vq.mid, vq.tte, vq.right, vq.iv
        fwd._S0 = spot_price  # Override the placeholder with the true spot_price.
        model._forwardCurve = fwd  # Since model takes the fwd curve, we must also update it.
        pricer._model = model  # Since pricer takes the model, we must also update it.

        is_calls = np.where(flags == "c", True, False)  # Array of booleans indicating call/puts.

        if s_initial_guess is not None:
            calibrator.set_initial_guess(
                s_initial_guess
            )  # Set the initial guess for the calibrator.

        try:
            # This is all fypy stuff
            calibrator.add_objective(
                "Targets",
                Targets(prices, lambda: pricer.price_strikes(T=ttm, K=strike, is_calls=is_calls)),
            )
            calibrator._model = model
            result = calibrator.calibrate()
            params = result.params
            model.set_params(params)

            fitted_prices = pricer.price_strikes(T=ttm, K=strike, is_calls=is_calls)
            fitted_iv = viv(
                fitted_prices,
                spot_price,
                strike,
                ttm,
                0.05,
                flags,
                return_as="numpy",
                on_error="ignore",
            )
            s_initial_guess = params
            s_fitted_iv = fitted_iv
            return VolFit(
                model_name="kou_de",
                params=params,
                strike=strike,
                iv=iv,
                fitted_iv=fitted_iv,
            )
        except Exception as e:
            logging.error("ERROR when fitting Kou DE model: %s", e)
            # If fitting fails, use the original IV as fitted_iv fallback
            fallback_fitted_iv = iv if s_fitted_iv is None else s_fitted_iv
            return VolFit(
                model_name="kou_de",
                params=s_initial_guess,
                strike=strike,
                iv=iv,
                fitted_iv=fallback_fitted_iv,
            )
