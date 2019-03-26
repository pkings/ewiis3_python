import time

from customer_demand_predictor import util
from customer_demand_predictor.predictors import Sarima
import ewiis3DatabaseConnector as data

def train_production_model(df_total_grid_consumption_and_production, game_id):
    sarima_predictor = Sarima()
    sarima_predictor.train_SARIMA_model(df_total_grid_consumption_and_production, game_id, 'totalProduction', ['cloudCover', 'temperature'], 'grid', 'production', 'SARIMAX')


def predict_production(df_total_grid_consumption_and_production, game_id):
    sarima_predictor = Sarima()
    df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(df_total_grid_consumption_and_production, game_id, 'totalProduction', ['cloudCover', 'temperature'], 'grid', 'production', 'SARIMAX')
    data.store_predictions(df_prediction, 'prosumption_prediction')


def train_consumption_model(df_total_grid_consumption_and_production, game_id):
    sarima_predictor = Sarima()
    sarima_predictor.train_SARIMA_model(df_total_grid_consumption_and_production, game_id, 'totalConsumption', ['temperature'], 'grid', 'consumption', 'SARIMAX') # ['isWeekend', 'temperature'] causes trouble with nan values


def predict_consumption(df_total_grid_consumption_and_production, game_id):
    sarima_predictor = Sarima()
    df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(df_total_grid_consumption_and_production, game_id, 'totalConsumption', ['temperature'], 'grid', 'consumption', 'SARIMAX')
    data.store_predictions(df_prediction, 'prosumption_prediction')


def train_all_predictors(df_total_grid_consumption_and_production, game_id):
    train_production_model(df_total_grid_consumption_and_production, game_id)
    train_consumption_model(df_total_grid_consumption_and_production, game_id)
    print('Successfully trained models for consumption and production.')


def predict_prosumption(df_total_grid_consumption_and_production, game_id):
    predict_production(df_total_grid_consumption_and_production, game_id)
    predict_consumption(df_total_grid_consumption_and_production, game_id)
    print('Successfully predicted consumption and production.')


def train_and_predict_all(df_total_grid_consumption_and_production, game_id):
    train_all_predictors(df_total_grid_consumption_and_production, game_id)
    predict_prosumption(df_total_grid_consumption_and_production, game_id)


def check_for_existing_prediction(df_total_grid_consumption_and_production, game_id):
    latest_timeslot = max(df_total_grid_consumption_and_production['timeslot'])
    df_prosumption_prediction = data.load_predictions('prosumption_prediction', game_id)

    if df_prosumption_prediction.empty:
        return False

    latest_prediction_timeslot = max(df_prosumption_prediction['prediction_timeslot'])
    return True if latest_timeslot == latest_prediction_timeslot else False


if __name__ == '__main__':
    while True:
        retrain_models = 20
        try:
            start_time = time.time()
            current_game_id, latest_timeslot = data.get_current_game_id_and_timeslot()
            df_total_grid_consumption_and_production, game_id = data.load_grid_consumption_and_production(current_game_id)

            if df_total_grid_consumption_and_production.empty:
                print('No data available yet.')
            elif len(df_total_grid_consumption_and_production['timeslot'].unique()) <= 24:
                print('Not enough data to build models and predict')
            elif check_for_existing_prediction(df_total_grid_consumption_and_production, game_id):
                print('Predictions have already be stores for the current available data.')
            elif util.check_for_model_existence(util.build_model_save_path('grid', 'consumption', 'SARIMAX')) and not len(df_total_grid_consumption_and_production['timeslot'].unique()) % retrain_models == 0:  # TODO: must check for all models, not only one
                predict_prosumption(df_total_grid_consumption_and_production, game_id)
            else:
                train_and_predict_all(df_total_grid_consumption_and_production, game_id)
            print('Total grid prosumption prediction (and training) lasted {} seconds'.format(time.time() - start_time))
        except Exception as e:
            print("ERROR: some error has occurred during iteration.")
            print(e)
        time.sleep(1.5)
