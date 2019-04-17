import time
import pandas as pd

from ewiis3_python_scripts import util
from ewiis3_python_scripts.predictors import Sarima, PredictorAbstract
import ewiis3DatabaseConnector as data

class CustomerProsumptionPredictor(PredictorAbstract):
    order = (1, 0, 0)
    seasonal_order = None
    forecast_length = 24
    min_observations = 2
    current_game_id = None
    latest_timeslot = None
    df_customer_prosumption = pd.DataFrame()
    df_weather_forecast = pd.DataFrame()
    endog_var = 'SUM(kWH)'
    exog_vars = ['cloudCover', 'temperature', 'windSpeed']
    target = 'customer'
    type = 'prosumption'


    def __init__(self, game_id):
        self.current_game_id = game_id
        pass


    def load_data(self):
        self.latest_timeslot = data.load_latest_timeslot_of_gameId(self.current_game_id)
        df_customer_prosumption, game_id = data.load_customer_prosumption_with_weather_and_time(self.current_game_id)
        df_customer_prosumption.rename(columns={'timeslotIndex': 'timeslot'}, inplace=True)
        self.df_customer_prosumption = df_customer_prosumption
        df_weather_forecast, game_id = data.load_weather_forecast(self.current_game_id)
        self.df_weather_forecast = df_weather_forecast[self.exog_vars]
        pass


    def set_SARIMA_Parameter(self):
        pass


    def train(self):
        sarima_predictor = Sarima()
        sarima_predictor.set_SARIMA_Parameter(order=self.order, seasonal_order=self.seasonal_order, forecast_length=self.forecast_length)
        sarima_predictor.train_SARIMA_model(self.df_customer_prosumption, self.current_game_id, self.endog_var, self.exog_vars, self.target, self.type, 'SARIMAX')
        print('Successfully (re)trained customer prosumption model.')


    def predict(self):
        sarima_predictor = Sarima()
        sarima_predictor.set_SARIMA_Parameter(order=self.order, seasonal_order=self.seasonal_order, forecast_length=self.forecast_length)
        df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(self.df_customer_prosumption, self.current_game_id, self.endog_var, self.exog_vars, self.df_weather_forecast, self.target, self.type, 'SARIMAX')
        data.store_predictions(df_prediction, 'prediction')
        print('Successfully predicted customer prosumption.')


    def check_for_existing_prediction(self):
        latest_timeslot = max(self.df_customer_prosumption['timeslot'])
        df_customer_prosumption_predicton = data.load_predictions('prediction', self.current_game_id, 'customer', 'prosumption')

        if df_customer_prosumption_predicton.empty:
            return False

        latest_prediction_timeslot = max(df_customer_prosumption_predicton['prediction_timeslot'])
        return True if latest_timeslot == latest_prediction_timeslot else False


    def check_for_model_existence(self):
        return util.check_for_model_existence(util.build_model_save_path(self.current_game_id, self.target, self.type, 'SARIMAX'))


    def get_size_of_training_data(self):
        return len(self.df_customer_prosumption['timeslot'].unique())


    def has_enough_observations_for_training(self):
        return self.get_size_of_training_data() > self.min_observations


def process_gameId(game_id):
    try:
        start_time = time.time()
        retrain_models = 20

        customerProsumptionPredictor = CustomerProsumptionPredictor(game_id)
        customerProsumptionPredictor.load_data()

        if not customerProsumptionPredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
        else:

            # switch to saisonal model
            if customerProsumptionPredictor.get_size_of_training_data() > 40 and customerProsumptionPredictor.seasonal_order is None:
                customerProsumptionPredictor.seasonal_order = (1, 0, 0, 24)
                customerProsumptionPredictor.train()

            # training model
            if not customerProsumptionPredictor.check_for_model_existence() or (
                                customerProsumptionPredictor.get_size_of_training_data() % retrain_models == 0 and customerProsumptionPredictor.get_size_of_training_data() > 30):
                customerProsumptionPredictor.train()
            # predict
            if not customerProsumptionPredictor.check_for_existing_prediction():
                customerProsumptionPredictor.predict()
        print('Customer prosumption prediction (and training) lasted {} seconds'.format(time.time() - start_time))
    except Exception as e:
        print("ERROR: some error has occurred during iteration.")
        print(e)


def process_run():
    for game_id in data.get_running_gameIds():
        process_gameId(game_id)


if __name__ == '__main__':
    while True:
        process_run()
        time.sleep(1.5)
