import pandas as pd
import time

from customer_demand_predictor import data, util
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX

order = (1, 0, 0)
seasonal_order = (1, 0, 0, 24)
forecast_length = 24


def train_SARIMA_model(dep_var, indep_var, target, type, model_type):
    df_total_grid_consumption_and_production, game_id = data.load_total_grid_consumption_and_production()

    Y = df_total_grid_consumption_and_production[dep_var]
    X = df_total_grid_consumption_and_production[indep_var]

    if len(Y) <= 24:
        print("Length of dependent variable is to short: {}".format(len(Y)))
        return

    if len(X) <= 24:
        print("Length of dependent variables is to short: {}".format(len(X)))
        return

    model = SARIMAX(endog=Y, exog=X, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted_model = model.fit()
    # print(fitted_model.summary())
    save_path = util.build_model_save_path(target, type, model_type)
    fitted_model.save(save_path, remove_data=True)


def predict_with_trained_SARIMA_model(dep_var, indep_var, target, type, model_type):
    # might be mistakes here!
    save_path = util.build_model_save_path(target, type, model_type)

    if not util.check_for_model_existence(save_path):
        print("Cannot predict for {}. No model has been trained yet.".format(dep_var))
        return

    saved_model = SARIMAXResults.load(save_path)
    df_total_grid_consumption_and_production, game_id = data.load_total_grid_consumption_and_production()

    Y = df_total_grid_consumption_and_production[dep_var]
    X = df_total_grid_consumption_and_production[indep_var]
    latest_index = max(Y.index)
    latest_timeslot = df_total_grid_consumption_and_production.iloc[latest_index]['timeslot']
    model = SARIMAX(endog=Y, exog=X, order=order, seasonal_order=seasonal_order)
    fitted_model = model.filter(params=saved_model.params)
    prediction = fitted_model.predict(exog=X[-24:], start=latest_index + 1, end=latest_index + 1 + forecast_length - 1)
    df_prediction = pd.DataFrame({'target_timeslot': range(latest_timeslot + 1, latest_timeslot + 1 + forecast_length), 'prediction': prediction})
    df_prediction['prediction_timeslot'] = latest_timeslot
    df_prediction['proximity'] = df_prediction['target_timeslot'] - df_prediction['prediction_timeslot']
    df_prediction['game_id'] = game_id

    df_prediction['target'] = target
    df_prediction['type'] = type
    data.store_prosumption_predictions(df_prediction)


def train_production_model():
    train_SARIMA_model('totalProduction', ['cloudCover', 'temperature'], 'grid', 'production', 'SARIMAX')


def predict_production():
    predict_with_trained_SARIMA_model('totalProduction', ['cloudCover', 'temperature'], 'grid', 'production', 'SARIMAX')


def train_consumption_model():
    train_SARIMA_model('totalConsumption', ['cloudCover', 'temperature'], 'grid', 'consumption', 'SARIMAX')


def predict_consumption():
    predict_with_trained_SARIMA_model('totalConsumption', ['cloudCover', 'temperature'], 'grid', 'consumption', 'SARIMAX')


def train_all_predictors():
    train_production_model()
    train_consumption_model()
    print('Successfully trained models for consumption and production.')


def predict_prosumption():
    predict_production()
    predict_consumption()
    print('Successfully predicted consumption and production.')


def train_and_predict_all():
    train_all_predictors()
    predict_prosumption()


def check_for_existing_prediction(df_total_grid_consumption_and_production):
    latest_timeslot = max(df_total_grid_consumption_and_production['timeslot'])
    df_prosumption_prediction = data.load_prosumption_predictions()

    if df_prosumption_prediction.empty:
        return False

    latest_prediction_timeslot = max(df_prosumption_prediction['prediction_timeslot'])
    return True if latest_timeslot == latest_prediction_timeslot else False


if __name__ == '__main__':
    while True:
        retrain_models = 5
        try:
            start_time = time.time()
            df_total_grid_consumption_and_production, game_id = data.load_total_grid_consumption_and_production()

            if df_total_grid_consumption_and_production.empty:
                print('No data available yet.')
            elif len(df_total_grid_consumption_and_production['timeslot'].unique()) <= 24:
                print('Not enough data to build models and predict')
            elif check_for_existing_prediction(df_total_grid_consumption_and_production):
                print('Predictions have already be stores for the current available data.')
            elif util.check_for_model_existence(util.build_model_save_path('grid', 'consumption', 'SARIMAX')) and not len(df_total_grid_consumption_and_production['timeslot'].unique()) % retrain_models == 0:  # TODO: must check for all models, not only one
                predict_prosumption()
            else:
                train_and_predict_all()
            print('Total grid prosumption prediction (and training) lasted {} seconds'.format(time.time() - start_time))
        except Exception as e:
            print("ERROR: some error has occurred during iteration.")
            print(e)
        time.sleep(2)
