
import logging

import numpy as np
import numpy.ma as ma

logging.basicConfig(format="%(levelname)s: %(module)s.%(funcName)s(): %(message)s")

def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed) ** 2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan


def lognashsutcliffe(evaluation, simulation, epsilon=0):
    """
    log Nash-Sutcliffe model efficiency
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :epsilon: Value which is added to simulation and evaluation data to errors when simulation or evaluation data has zero values
    :type: float or list
    :return: log Nash-Sutcliffe model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation) + epsilon, np.array(evaluation) + epsilon
        return float(
            1- np.nansum((np.log(s) - np.log(e)) ** 2)/ np.nansum((np.log(e) - np.nanmean(np.log(e))) ** 2)
        )
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan
      
def nse_FDC(evaluation, simulation, return_all=False):
  
    if len(evaluation) == len(simulation):

        fdc_sim = np.sort(simulation / (np.nanmean(simulation) * len(simulation)))
        fdc_obs = np.sort(evaluation / (np.nanmean(evaluation) * len(evaluation)))

        mean_observed = np.nanmean(fdc_obs)
        # compute numerator and denominator
        numerator = np.nansum(( fdc_obs -  fdc_sim) ** 2)
        denominator = np.nansum((fdc_obs - mean_observed) ** 2)
        return 1 - (numerator / denominator)
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan




def rsquared(evaluation, simulation):
    """
    Coefficient of Determination
        .. math::
         r^2=(\\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})^2
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Coefficient of Determination
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        return correlationcoefficient(evaluation, simulation) ** 2
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan


def mse(evaluation, simulation):
    """
    Mean Squared Error
        .. math::
         MSE=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Mean Squared Error
    :rtype: float
    """

    if len(evaluation) == len(simulation):
        obs, sim = np.array(evaluation), np.array(simulation)
        mse = np.nanmean((obs - sim) ** 2)
        return mse
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan


def rmse(evaluation, simulation):
    """
    Root Mean Squared Error
        .. math::
         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Root Mean Squared Error
    :rtype: float
    """
    if len(evaluation) == len(simulation) > 0:
        return np.sqrt(mse(evaluation, simulation))
    else:
        logging.warning("evaluation and simulation lists do not have the same length.")
        return np.nan


def mae(evaluation, simulation):
    """
    Mean Absolute Error
        .. math::
         MAE=\\frac{1}{N}\\sum_{i=1}^{N}(\\left |  e_{i}-s_{i} \\right |)
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Mean Absolute Error
    :rtype: float
    """
    if len(evaluation) == len(simulation) > 0:
        obs, sim = np.array(evaluation), np.array(simulation)
        mae = np.nanmean(np.abs(sim - obs))
        return mae
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan


def rrmse(evaluation, simulation):
    """
    Relative Root Mean Squared Error
        .. math::
         RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Relative Root Mean Squared Error
    :rtype: float
    """

    if len(evaluation) == len(simulation):
        rrmse = rmse(evaluation, simulation) / np.nanmean(evaluation)
        return rrmse
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan




def kge(evaluation, simulation, return_all=False):
    """
    Kling-Gupta Efficiency
    Corresponding paper:
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling
    output:
        kge: Kling-Gupta Efficiency
    optional_output:
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    if len(evaluation) == len(simulation):
        cc =  ma.corrcoef(ma.masked_invalid(evaluation), ma.masked_invalid(simulation))[0, 1]
        alpha = np.nanstd(simulation) / np.nanstd(evaluation)
        beta = np.nansum(simulation) / np.nansum(evaluation)
        kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan


def volume_error(evaluation, simulation):
    """
    Returns the Volume Error (Ve).
    It is an indicator of the agreement between the averages of the simulated
    and observed runoff (i.e. long-term water balance).
    used in this paper:
    Reynolds, J.E., S. Halldin, C.Y. Xu, J. Seibert, and A. Kauffeldt. 2017.
    “Sub-Daily Runoff Predictions Using Parameters Calibrated on the Basis of Data with a
    Daily Temporal Resolution.” Journal of Hydrology 550 (July):399–411.
    https://doi.org/10.1016/j.jhydrol.2017.05.012.
        .. math::
         Sum(simulation-evaluation)/sum(simulation)
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Volume Error
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        ve = np.nansum(simulation - evaluation) / np.nansum(evaluation)
        return float(ve)
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan


_all_functions = [
    kge,
    lognashsutcliffe,
    mae,
    mse,
    nashsutcliffe,
    rmse,
    rrmse,
    volume_error
]


def calculate_all_functions(evaluation, simulation):
    """
    Calculates all objective functions from spotpy.objectivefunctions
    and returns the results as a list of name/value pairs
    :param evaluation: a sequence of evaluation data
    :param simulation: a sequence of simulation data
    :return: A list of (name, value) tuples
    """

    result = []
    for f in _all_functions:
        # Check if the name is not private and attr is a function but not this

        try:
            result.append((f.__name__, f(evaluation, simulation)))
        except:
            result.append((f.__name__, np.nan))

    return result
