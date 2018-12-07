import math
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from customer_demand_predictor.sarima import mean_percentage_error, total_deviation_error, calcuate_R_2


def predict_with_sarimax(timeSeries, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24), forecast_length=24):
    # specify explicit lags: iterable of booleans [0,1,0,0,1]
    y_train = timeSeries[:-forecast_length]
    y_test = timeSeries[-forecast_length:]

    if seasonal_order == None:
        model = SARIMAX(endog=y_train, order=order, enforce_stationarity=False,
                        enforce_invertibility=False)
    else:
        model = SARIMAX(endog=y_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

    fitted_model = model.fit()
    prediction = fitted_model.predict(start=len(y_train), end=len(y_train) + forecast_length - 1)

    prediction.index = y_test.index
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = math.sqrt(mse)
    # mpe = mean_percentage_error(y_test, prediction)
    # absolute_deviation = total_deviation_error(y_test, prediction)
    rsquared = calcuate_R_2(prediction, y_test)
    return prediction, mae, mse, rmse, fitted_model.aic, fitted_model.bic, fitted_model.resid, rsquared


def find_best_sarima_model_params(ts, forecast_length, p_max, q_max, sp_max=0, sq_max=0):
    best_models = {
        'best_aic': {'order': (), 'seasonal_order': (), 'mae': sys.maxsize, 'bic': sys.maxsize, 'aic': sys.maxsize},
        'best_bic': {'order': (), 'seasonal_order': (), 'mae': sys.maxsize, 'bic': sys.maxsize, 'aic': sys.maxsize}
    }

    for p in range(0, p_max+1):
        for q in range(0, q_max+1):
            for sp in range(0, sp_max+1):
                for sq in range(0, sq_max+1):
                    if p == 0 and q == 0:
                        continue
                    order = (p, 0, q)
                    seasonal_order = (sp, 0, sq, 24)
                    try:
                        prediction, mae, mse, rmse, aic, bic, resid, rsquared = predict_with_sarimax(ts,
                                                                                                     order=order,
                                                                                                     seasonal_order=seasonal_order,
                                                                                                     forecast_length=forecast_length)
                        current_cs = {'aic': aic, 'bic': bic}
                        ICs = ['bic', 'aic']
                        for ic in ICs:
                            if (current_cs[ic] < best_models['best_{}'.format(ic)][ic]):
                                best_models['best_{}'.format(ic)]['order'] = order
                                best_models['best_{}'.format(ic)]['seasonal_order'] = seasonal_order
                                best_models['best_{}'.format(ic)]['mae'] = mae
                                best_models['best_{}'.format(ic)]['mse'] = mse
                                best_models['best_{}'.format(ic)]['rmse'] = rmse
                                best_models['best_{}'.format(ic)]['rsquared'] = rsquared
                                best_models['best_{}'.format(ic)]['bic'] = bic
                                best_models['best_{}'.format(ic)]['aic'] = aic
                    except Exception as e:
                        print('error')
    return best_models