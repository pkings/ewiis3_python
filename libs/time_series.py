from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import logging
import numpy as np
from numpy import polyfit
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX
from statsmodels.tsa.x13 import x13_arima_select_order, _find_x12
from libs.ts_clustering import ts_cluster
import statsmodels.api as sm



class TimeSeries:
    def __init__(self, time_series, prediction_length, test_split, model_dir, sarima_models_suffix, figure_output_dir='figures', name=None):
        self.raw_time_series = time_series.dropna()
        self.name = name
        self.train = self.raw_time_series.values
        self.prediction_length = prediction_length
        self.test = None
        self.test_split = test_split
        if test_split:
            self.split_to_train_and_test(prediction_length)
        self.prediction = None
        self.seasonal_adjustment_coefficients = None
        self.seasonal_adjustment_degree = 5
        self.prediction_errors = {}
        self.model_dir = model_dir
        self.sarima_models_suffix = sarima_models_suffix
        self.figure_output_dir = figure_output_dir


    ####################################################
    #### Autoregression
    ####################################################

    def predict_with_autoregression(self):
        # train autoregression
        model = AR(self.train)
        model_fit = model.fit()
        # make predictions
        self.prediction = model_fit.predict(start=len(self.train), end=len(self.train) + self.prediction_length - 1, dynamic=False)
        return self.prediction

    ####################################################
    #### Seasonal adjusted Autoregression
    ####################################################

    def predict_with_seasonal_adjustment(self):
        self.__do_seasonal_adjustment(self.figure_output_dir, self.name)
        predicion_of_residuals = self.predict_with_autoregression()
        seasonal_component = self.__get_seasonal_component_for_forecast()
        self.prediction = predicion_of_residuals + seasonal_component
        return self.prediction

    def __do_seasonal_adjustment(self, figure_dir, filename, createPlots=False):
        # fit polynomial: x^2*b1 + x*b2 + ... + bn
        X = [i % 24 for i in range(0, len(self.train))]
        y = self.train
        # degree = self.__find_best_degree(X, y) # das bring fast nichts, dauert aber sehr lange
        self.seasonal_adjustment_coefficients = polyfit(X, y, self.seasonal_adjustment_degree)
        # create curve
        curve = list()
        for i in range(len(X)):
            curve.append(self.__calculate_value_of_random_component(X[i]))

        ##########################
        ''' We can now use this model to create a seasonally adjusted version of the dataset. '''
        ##########################

        # ... previous code ...
        # create seasonally adjusted
        values = self.train
        diff = list()
        for i in range(len(values)):
            value = values[i] - curve[i]
            diff.append(value)
        if createPlots:
            fig, ax = plt.subplots()
            plt.plot(diff)
            plt.savefig('{}{}_seasonal_residuals.png'.format(figure_dir, filename.replace(' ', '_')))
        self.train = pd.Series(diff).values
        return curve

    def __calculate_value_of_random_component(self, x):
        value = self.seasonal_adjustment_coefficients[-1]
        for d in range(self.seasonal_adjustment_degree):
            value += x ** (self.seasonal_adjustment_degree - d) * self.seasonal_adjustment_coefficients[d]
        return value


    def __get_seasonal_component_for_forecast(self):
        X = [i % 24 for i in range(len(self.train), len(self.train) + self.prediction_length)]
        curve = list()
        for i in range(len(X)):
            curve.append(self.__calculate_value_of_random_component(X[i]))
        return np.array(curve)

    ####################################################
    #### SARIMA
    ####################################################

    def predict_with_sarima(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24)):
        # seasonal_order=(1, 1, 1, 24),
        saved_model = SARIMAXResults.load('{}{}{}.pkl'.format(self.model_dir, self.name, self.sarima_models_suffix))
        data_used_for_prediction = self.train[-48:]
        model = SARIMAX(endog=data_used_for_prediction, order=order, seasonal_order=seasonal_order)
        fitted_model = model.filter(params=saved_model.params)
        self.prediction = fitted_model.predict(start=len(data_used_for_prediction), end=len(data_used_for_prediction) + self.prediction_length - 1)
        return self.prediction

    def train_sarima_model(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24), save_model=False):
        # best for brookerside: order=(2, 1, 2), seasonal_order=(2, 0, 0, 24),
        # seasonal_order=(1, 1, 1, 24),
        if adfuller(self.train)[1] > 0.05:
            logging.warning('We cannot assume stationarity for customer {}'.format(self.name))

        # result_acf, qstat, pvalue = acf(self.train, qstat=True)
        # result_acf, qstat, pvalue = pacf(self.train, qstat=True)
        # self.create_acf_and_pacf_plot(on_differences=True)
        if seasonal_order == None:
            model = SARIMAX(endog=self.train, order=order, enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = SARIMAX(endog=self.train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        # param_result = x13_arima_select_order(self.train, x12path='/Users/peterkings/Downloads/x13asall_V1.1_B39/x13as/x13as')
        # print(param_result.order, param_result.sorder)
        fitted_model = model.fit()
        if save_model:
            fitted_model.save('{}{}{}.pkl'.format(self.model_dir, self.name, self.sarima_models_suffix), remove_data=True)
        if self.test_split:
            data_used_for_prediction = self.train[-48:]
            self.prediction = fitted_model.predict(start=len(data_used_for_prediction),
                                               end=len(data_used_for_prediction) + self.prediction_length - 1)
        return fitted_model.params

    ####################################################
    #### Helpers
    ####################################################

    ##########################
    ''' Train/Test Split '''
    ##########################

    def split_to_train_and_test(self, size_of_test_data):
        X = self.raw_time_series.values
        train, test = X[:len(X) - size_of_test_data], X[len(X) - size_of_test_data:]
        self.train = train
        self.test = test

    def reset_train_test_split(self):
        self.train = self.raw_time_series.values
        self.test = None
        self.prediction = None

    ##########################
    ''' Prediction Errors '''
    ##########################

    def calculate_prediction_errors(self):
        if self.test_split:
            self.prediction_errors['mae'] = mean_absolute_error(self.test, self.prediction)
            self.prediction_errors['mse'] = mean_squared_error(self.test, self.prediction)
            return self.prediction_errors
        return 'no train / test split'

    def __mean_percentage_error(self, y_true, y_predict):
        output = np.average(abs(y_true - y_predict) / y_true) * 100
        return output

    def __total_deviation_error(self, y_true, y_predict):
        return np.sum(abs(y_true - y_predict))

    ##########################
    ''' Time Series Clustering '''
    ##########################

    def cluster_time_serieses(self, time_serieses, num_custer=10):
        for i in range(1, num_custer+1):
            ts_c = ts_cluster(time_serieses, i)
            ts_c.k_means_clust(4, 10)
            ts_c.plot_centroids()
            ts_c.plot_cluster_assigments()

    ####################################################
    #### Plot Helpers
    ####################################################

    def plot_prediction(self):
        # plot results
        fig, ax = plt.subplots()
        actual_values = self.raw_time_series.values
        prediction_values = np.array([None for i in range(len(self.train))])
        prediction_values = np.append(prediction_values, self.prediction)
        plt.plot(actual_values, color='#14779b', label='actual')
        plt.plot(prediction_values, color='#e8483b', label='prediction')
        legend = ax.legend(loc='lower right')
        legend.get_frame().set_facecolor('w')
        plt.savefig('{}{}_prediction'.format(self.figure_output_dir, self.name))

    def create_time_series_plot(self, filename):
        fig, ax = plt.subplots()
        self.raw_time_series.plot()
        plt.savefig('{}{}_time_series.png'.format(self.figure_output_dir, filename))
        pass

    def create_lag_plot(self, filename):
        fig, ax = plt.subplots()
        lag_plot(self.raw_time_series)
        plt.savefig('{}{}_lag_plot.png'.format(self.figure_output_dir, filename))
        pass

    def get_correlation_matrix(self):
        values = pd.DataFrame(self.raw_time_series.values)
        dataframe = pd.concat([values.shift(1), values], axis=1)
        dataframe.columns = ['t-1', 't+1']
        result = dataframe.corr()
        return result

    def create_autocorrelation_plot(self, filename):
        fig, ax = plt.subplots()
        autocorrelation_plot(self.raw_time_series)
        plt.savefig('{}/{}_autocorrelation_plot.png'.format(self.figure_output_dir, filename))

    def create_acf_and_pacf_plot(self, on_differences=False):
        x = self.raw_time_series.diff().dropna() if on_differences else self.raw_time_series
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(x, lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(x, lags=40, ax=ax2)
        plt.savefig('{}/{}_acf_and_pacf_plot.png'.format(self.figure_output_dir, self.name))
