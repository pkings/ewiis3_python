import time
import pandas as pd

from customer_demand_predictor import util
from customer_demand_predictor.predictors import Sarima, PredictorAbstract
import ewiis3DatabaseConnector as data


class ImbalancePredictor(PredictorAbstract):
    order = (1, 0, 0)
    seasonal_order = None
    forecast_length = 24
    min_observations = 2
    current_game_id = None
    latest_timeslot = None
    df_grid_imbalance = pd.DataFrame()
    df_weather_forecast = pd.DataFrame()
    endog_var = 'netImbalance'
    exog_vars = ['cloudCover', 'temperature', 'windSpeed']
    target = 'grid'
    type = 'imbalance'


    def __init__(self):
        pass


    def load_data(self):
        self.current_game_id, self.latest_timeslot = data.get_current_game_id_and_timeslot()
        df_grid_imbalance, game_id = data.load_grid_imbalance(self.current_game_id)
        df_grid_imbalance.rename(columns={'timeslotIndex': 'timeslot'}, inplace=True)
        self.df_grid_imbalance = df_grid_imbalance
        df_weather_forecast, game_id = data.load_weather_forecast(self.current_game_id)
        self.df_weather_forecast = df_weather_forecast[self.exog_vars]


    def set_SARIMA_Parameter(self, order, seasonal_order, forecast_length):
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_length = forecast_length


    def train(self):
        sarima_predictor = Sarima()
        sarima_predictor.set_SARIMA_Parameter(order=self.order, seasonal_order=self.seasonal_order, forecast_length=self.forecast_length)
        sarima_predictor.train_SARIMA_model(self.df_grid_imbalance, self.current_game_id, self.endog_var, self.exog_vars, self.target, self.type, 'SARIMAX')
        print('Successfully (re)trained grid imbalance model.')


    def predict(self):
        sarima_predictor = Sarima()
        sarima_predictor.set_SARIMA_Parameter(order=self.order, seasonal_order=self.seasonal_order, forecast_length=self.forecast_length)
        df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(self.df_grid_imbalance, self.current_game_id, self.endog_var, self.exog_vars, self.df_weather_forecast, self.target, self.type, 'SARIMAX')
        data.store_predictions(df_prediction, 'prediction')
        print('Successfully predicted grid imbalance.')


    def check_for_existing_prediction(self):
        latest_timeslot = max(self.df_grid_imbalance['timeslot'])
        df_imbalance_prediction = data.load_predictions('prediction', self.current_game_id, 'grid', 'imbalance')

        if df_imbalance_prediction.empty:
            return False

        latest_prediction_timeslot = max(df_imbalance_prediction['prediction_timeslot'])
        return True if latest_timeslot == latest_prediction_timeslot else False


    def check_for_model_existence(self):
        return util.check_for_model_existence(util.build_model_save_path(self.target, self.type, 'SARIMAX'))


    def get_size_of_training_data(self):
        return len(self.df_grid_imbalance['timeslot'].unique())


    def has_enough_observations_for_training(self):
        return self.get_size_of_training_data() > self.min_observations



if __name__ == '__main__':
    while True:
        retrain_models = 20
        try:
            start_time = time.time()

            imbalancePredictor = ImbalancePredictor()
            imbalancePredictor.load_data()

            if not imbalancePredictor.has_enough_observations_for_training():
                print('Not enough data to build models and predict')
            else:
                # training model
                if not imbalancePredictor.check_for_model_existence() or (imbalancePredictor.get_size_of_training_data() % retrain_models == 0 and imbalancePredictor.get_size_of_training_data() > 30):
                    imbalancePredictor.train()
                # predict
                if not imbalancePredictor.check_for_existing_prediction():
                    imbalancePredictor.predict()
            print('Grid imbalance prediction (and training) lasted {} seconds'.format(time.time() - start_time))
        except Exception as e:
            print("ERROR: some error has occurred during iteration.")
            print(e)
        time.sleep(1.5)
