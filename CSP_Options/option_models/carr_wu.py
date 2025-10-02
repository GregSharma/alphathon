import numpy as np
import csp
from csp import ts
from CSP_Options.structs import VectorizedOptionQuote, VolFit
from scipy.optimize import least_squares
import logging


def cw_model(theta, k, sigma, tau):
    v, m, w, eta, rho = theta
    return (
        1 / 4 * np.exp(-2 * eta * tau) * w**2 * tau**2 * sigma**4
        + (1 - 2 * np.exp(-eta * tau) * m * tau - np.exp(-eta * tau) * w * rho * np.sqrt(v) * tau)
        * sigma**2
        - (
            v
            + 2 * np.exp(-eta * tau) * w * rho * np.sqrt(v) * k
            + np.exp(-2 * eta * tau) * w**2 * k**2
        )
    )


def cw_residuals(theta, k, sigma, tau):
    return cw_model(theta, k, sigma, tau)


def cw_fit(strikes, sigmas, tte, initial_guess=[0.2, 0.04, 0.1, 0.1, 0]):
    def cw_objective(theta):
        return cw_residuals(theta, strikes, sigmas, tte)

    # Set bounds for the parameters
    lower_bounds = [0, -np.inf, 0, 0, -1]  # v > 0, m unbounded, w > 0, eta > 0, rho >= -1
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, 1]  # v, m, w, eta unbounded above, rho <= 1

    # Ensure initial guess is within bounds
    initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)

    result = least_squares(cw_objective, initial_guess, bounds=(lower_bounds, upper_bounds))
    return result.x


def cw_surface(strikes, theta, tte):
    v, m, w, eta, rho = theta

    # We need to solve a quadratic equation a*x^2 + b*x + c = 0
    a = 1 / 4 * np.exp(-2 * eta * tte) * w**2 * tte**2
    b = 1 - 2 * np.exp(-eta * tte) * m * tte - np.exp(-eta * tte) * w * rho * np.sqrt(v) * tte
    c = -(
        v
        + 2 * np.exp(-eta * tte) * w * rho * np.sqrt(v) * strikes
        + np.exp(-2 * eta * tte) * w**2 * strikes**2
    )

    # Quadratic formula
    discriminant = b**2 - 4 * a * c
    sigma_sq = (-b + np.sqrt(discriminant)) / (2 * a)

    sigma = np.sqrt(sigma_sq)
    return np.where(sigma >= 100, np.nan, sigma)


@csp.node
def node_cw_fit(vq: ts[VectorizedOptionQuote]) -> ts[VolFit]:
    with csp.state():
        s_initial_guess = np.array([0.2, 0.04, 0.1, 0.1, 0])
        s_fitted_iv = None
    if csp.ticked(vq):
        try:
            params = cw_fit(vq.strike, vq.iv, vq.tte, s_initial_guess)

            s_initial_guess = np.array(params)
            s_fitted_iv = cw_surface(vq.strike, params, vq.tte)
            return VolFit(
                model_name="carr_wu",
                params=s_initial_guess,
                strike=vq.strike,
                iv=vq.iv,
                fitted_iv=s_fitted_iv,
            )
        except Exception as e:
            logging.error(f"ERROR when fitting Carr Wu model: {e}")
            return VolFit(
                model_name="carr_wu",
                params=s_initial_guess,
                strike=vq.strike,
                iv=vq.iv,
                fitted_iv=s_fitted_iv,
            )
