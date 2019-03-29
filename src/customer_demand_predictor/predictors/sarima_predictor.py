import pandas as pd

from customer_demand_predictor import util
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX


class Sarima:
    order = (1, 0, 0)
    seasonal_order = (1, 0, 0, 24)
    forecast_length = 24

    def __init__(self):
        pass

    def set_SARIMA_Parameter(self, order, seasonal_order, forecast_length):
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_length = forecast_length

    def train_SARIMA_model(self, df, game_id, dep_var, indep_var, target, type,
                           model_type):
        Y = df[dep_var]
        X = df[indep_var]

        if len(Y) <= 24 and not self.seasonal_order is None:
            print("Length of dependent variable is to short: {}".format(len(Y)))
            return

        if len(X) <= 24 and not self.seasonal_order is None:
            print("Length of dependent variables is to short: {}".format(len(X)))
            return

        if self.seasonal_order is None:
            model = SARIMAX(endog=Y, exog=X, order=self.order, enforce_stationarity=False,  enforce_invertibility=False)
        else:
            model = SARIMAX(endog=Y, exog=X, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False,  enforce_invertibility=False)

        fitted_model = model.fit()
        # print(fitted_model.summary())
        save_path = util.build_model_save_path(target, type, model_type)
        fitted_model.save(save_path, remove_data=True)

    def predict_with_trained_SARIMA_model(self, df, game_id, dep_var, indep_var, target,
                                          type, model_type):
        # might be mistakes here!
        save_path = util.build_model_save_path(target, type, model_type)

        if not util.check_for_model_existence(save_path):
            print("Cannot predict for {}. No model has been trained yet.".format(dep_var))
            return

        saved_model = SARIMAXResults.load(save_path)

        Y = df[dep_var]
        X = df[indep_var]
        latest_index = max(Y.index)
        latest_timeslot = df.iloc[latest_index]['timeslot']

        if self.seasonal_order is None:
            model = SARIMAX(endog=Y, exog=X, order=self.order)
        else:
            model = SARIMAX(endog=Y, exog=X, order=self.order, seasonal_order=self.seasonal_order)

        fitted_model = model.filter(params=saved_model.params)
        prediction = fitted_model.predict(exog=X[-24:], start=latest_index + 1,
                                          end=latest_index + 1 + self.forecast_length - 1)
        df_prediction = pd.DataFrame(
            {'target_timeslot': range(latest_timeslot + 1, latest_timeslot + 1 + self.forecast_length),
             'prediction': prediction})

        # set lower and upper bound for prediction
        prediction_upper_bound = max(Y) * 1.2
        prediction_lower_bound = min(Y) * 0.8
        # restrict prediction in lower and upper bound
        df_prediction['prediction'] = df_prediction['prediction'].apply(
            lambda x: max([min([x, prediction_upper_bound]), prediction_lower_bound]))

        df_prediction['prediction_timeslot'] = latest_timeslot
        df_prediction['proximity'] = df_prediction['target_timeslot'] - df_prediction['prediction_timeslot']
        df_prediction['game_id'] = game_id

        df_prediction['target'] = target
        df_prediction['type'] = type
        return df_prediction
