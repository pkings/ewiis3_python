import time

from customer_demand_predictor import util
from customer_demand_predictor.predictors import Sarima
import ewiis3DatabaseConnector as data


def train_customer_prosumption_model(df_grid_imbalance, game_id):
    sarima_predictor = Sarima()
    sarima_predictor.set_SARIMA_Parameter(order=(1, 0, 0), seasonal_order=None, forecast_length=24)
    sarima_predictor.train_SARIMA_model(df_grid_imbalance, game_id, 'SUM(kWH)', ['cloudCover', 'temperature', 'windSpeed'], 'customer', 'prosumption', 'SARIMAX')


def predict_customer_prosumption_model(df_grid_imbalance, game_id):
    sarima_predictor = Sarima()
    sarima_predictor.set_SARIMA_Parameter(order=(1, 0, 0), seasonal_order=None, forecast_length=24)
    df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(df_grid_imbalance, game_id, 'SUM(kWH)', ['cloudCover', 'temperature', 'windSpeed'], 'customer', 'prosumption', 'SARIMAX')
    data.store_predictions(df_prediction, 'customer_prosumption_prediction')


def check_for_existing_prediction(df_grid_imbalance, game_id):
    latest_timeslot = max(df_grid_imbalance['timeslot'])
    df_imbalance_prediction = data.load_predictions('customer_prosumption_prediction', game_id)

    if df_imbalance_prediction.empty:
        return False

    latest_prediction_timeslot = max(df_imbalance_prediction['prediction_timeslot'])
    return True if latest_timeslot == latest_prediction_timeslot else False


if __name__ == '__main__':
    while True:
        retrain_models = 20
        try:
            start_time = time.time()
            current_game_id, latest_timeslot = data.get_current_game_id_and_timeslot()
            df_customer_prosumption, game_id = data.load_customer_prosumption_with_weather_and_time(current_game_id)
            df_customer_prosumption.rename(columns={'timeslotIndex': 'timeslot'}, inplace=True)
            if df_customer_prosumption.empty:
                print('No data available yet.')
            elif len(df_customer_prosumption['timeslot'].unique()) <= 5:
                print('Not enough data to build models and predict')
            elif check_for_existing_prediction(df_customer_prosumption, game_id):
                print('Predictions have already be stored for the current available data.')
            elif util.check_for_model_existence(util.build_model_save_path('customer', 'prosumption', 'SARIMAX')) and not len(df_customer_prosumption['timeslot'].unique()) % retrain_models == 0:  # TODO: must check for all models, not only one
                predict_customer_prosumption_model(df_customer_prosumption, game_id)
            else:
                train_customer_prosumption_model(df_customer_prosumption, game_id)
                predict_customer_prosumption_model(df_customer_prosumption, game_id)
            print('customer prosumption prediction (and training) lasted {} seconds'.format(time.time() - start_time))
        except Exception as e:
            print("ERROR: some error has occurred during iteration.")
            print(e)
        time.sleep(1.5)
