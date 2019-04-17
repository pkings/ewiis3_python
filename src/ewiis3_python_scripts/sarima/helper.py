import statsmodels.api as sm
import numpy as np


def calcuate_R_2(y_hat, y_actual):
    try:
        x = sm.add_constant(y_hat)  # constant intercept term
        model = sm.OLS(y_actual, x)
        fitted = model.fit()
        return fitted.rsquared
    except Exception as e:
        print('Error occured during auxiliary regression for R^2.')
    return -1

def mean_percentage_error(y_true, y_predict):
    output = np.average(abs((y_true - y_predict) / y_true)) * 100
    return output


def total_deviation_error(y_true, y_predict):
    return np.sum(abs(y_true - y_predict))
