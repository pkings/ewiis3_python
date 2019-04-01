import time
import pandas as pd

from customer_demand_predictor import util
from customer_demand_predictor.predictors import Sarima, PredictorAbstract
import ewiis3DatabaseConnector as data


class GridProsumptionPredictor(PredictorAbstract):
    order = (1, 0, 0)
    seasonal_order = (1, 0, 0, 24)
    forecast_length = 24
    min_observations = 25
    current_game_id = None
    latest_timeslot = None
    df_total_grid_consumption_and_production = pd.DataFrame()
    df_weather_forecast = pd.DataFrame()
    endog_var_prod = 'totalProduction'
    endog_var_con = 'totalConsumption'
    exog_vars_prod = ['cloudCover', 'temperature']
    exog_vars_con = ['temperature']
    target = 'grid'
    type_prod = 'production'
    type_con = 'consumption'


    def __init__(self):
        pass


    def load_data(self):
        self.current_game_id, self.latest_timeslot = data.get_current_game_id_and_timeslot()
        self.df_total_grid_consumption_and_production, game_id = data.load_grid_consumption_and_production(self.current_game_id)

        self.df_weather_forecast, game_id = data.load_weather_forecast(self.current_game_id)


    def set_SARIMA_Parameter(self, order, seasonal_order, forecast_length):
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_length = forecast_length


    def train(self):
        # Production
        sarima_predictor = Sarima()
        sarima_predictor.set_SARIMA_Parameter(order=self.order, seasonal_order=self.seasonal_order, forecast_length=self.forecast_length)
        sarima_predictor.train_SARIMA_model(self.df_total_grid_consumption_and_production, self.current_game_id, self.endog_var_prod, self.exog_vars_prod, self.target, self.type_prod, 'SARIMAX')

        # Consumption
        sarima_predictor = Sarima()
        sarima_predictor.set_SARIMA_Parameter(order=self.order, seasonal_order=self.seasonal_order, forecast_length=self.forecast_length)
        sarima_predictor.train_SARIMA_model(self.df_total_grid_consumption_and_production, self.current_game_id, self.endog_var_con, self.exog_vars_con, self.target, self.type_con, 'SARIMAX')  # ['isWeekend', 'temperature'] causes trouble with nan values
        print('Successfully (re)trained grid production and consumption models.')


    def predict(self):
        # Production
        sarima_predictor = Sarima()
        df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(self.df_total_grid_consumption_and_production, self.current_game_id, self.endog_var_prod, self.exog_vars_prod, self.df_weather_forecast[self.exog_vars_prod], self.target, self.type_prod, 'SARIMAX')
        data.store_predictions(df_prediction, 'prediction')
        # Consumption
        sarima_predictor = Sarima()
        df_prediction = sarima_predictor.predict_with_trained_SARIMA_model(self.df_total_grid_consumption_and_production, self.current_game_id, self.endog_var_con, self.exog_vars_con, self.df_weather_forecast[self.exog_vars_con], self.target, self.type_con, 'SARIMAX')
        data.store_predictions(df_prediction, 'prediction')
        print('Successfully predicted grid production and consumption.')


    def check_for_existing_prediction(self):
        latest_timeslot = max(self.df_total_grid_consumption_and_production['timeslot'])
        df_prosumption_prediction = data.load_predictions('prediction', self.current_game_id, 'grid', self.type_con)  # TODO: not checking for production data here

        if df_prosumption_prediction.empty:
            return False

        latest_prediction_timeslot = max(df_prosumption_prediction['prediction_timeslot'])
        return True if latest_timeslot == latest_prediction_timeslot else False


    def check_for_model_existence(self):
        return util.check_for_model_existence(util.build_model_save_path(self.target, self.type_con, 'SARIMAX')) and util.check_for_model_existence(util.build_model_save_path(self.target, self.type_prod, 'SARIMAX'))


    def get_size_of_training_data(self):
        return len(self.df_total_grid_consumption_and_production['timeslot'].unique())


    def has_enough_observations_for_training(self):
        return self.get_size_of_training_data() > self.min_observations



if __name__ == '__main__':
    while True:
        retrain_models = 20
        try:
            start_time = time.time()

            gridProsumptionPredictor = GridProsumptionPredictor()
            gridProsumptionPredictor.load_data()


            if not gridProsumptionPredictor.has_enough_observations_for_training():
                print('Not enough data to build models and predict')
            else:
                # training model
                if not gridProsumptionPredictor.check_for_model_existence() or (gridProsumptionPredictor.get_size_of_training_data() % retrain_models == 0 and gridProsumptionPredictor.get_size_of_training_data() > 30):
                    gridProsumptionPredictor.train()
                # predict
                if not gridProsumptionPredictor.check_for_existing_prediction():
                    gridProsumptionPredictor.predict()
            print('Total grid prosumption prediction (and training) lasted {} seconds'.format(time.time() - start_time))
        except Exception as e:
            print("ERROR: some error has occurred during iteration.")
            print(e)
        time.sleep(1.5)
