from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


def create_rolling_mean_and_std_dev_plot(timeseries):
    decomposition = seasonal_decompose(timeseries, freq=24)
    fig = plt.figure()
    fig = decomposition.plot()
    fig.set_size_inches(15, 8)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # return trend, seasonal, residual


def perform_dickey_fuller_test(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def create_ACF_and_PACF_plot(time_series, save_path=''):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(time_series, lags=50, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(time_series, lags=50, ax=ax2)
    if save_path:
        plt.savefig(save_path)