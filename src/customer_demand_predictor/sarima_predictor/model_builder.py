import logging

import customer_demand_predictor as cdp
from customer_demand_predictor.sarima import find_best_sarima_model_params
from customer_demand_predictor.sarima.time_series import TimeSeries
from customer_demand_predictor.util import store_model_selection


def train_models_for_each_customer(df_consumption_and_production):
    customers_to_train_models = list(df_consumption_and_production['customerName'].unique())

    for customer in customers_to_train_models:
        df_customer = df_consumption_and_production[df_consumption_and_production['customerName'] == customer]
        series_customer = df_customer['kWh'][-96:]  # train on last 96 timesots

        best_model_params = find_best_sarima_model_params(ts=series_customer, forecast_length=24, p_max=2, q_max=2, sp_max=1, sq_max=1)

        logging.info('Best model params for customer {}:'.format(customer))
        logging.info(best_model_params)
        store_model_selection(best_model_params, customer)
        logging.info('Train model for customer {} with parameters {}.'.format(customer, best_model_params['best_aic']))

        time_series = TimeSeries(series_customer, 24, False, cdp.MODEL_DIR, cdp.SARIMA_MODEL_SUFFIX, name=customer)
        time_series.train_sarima_model(order=best_model_params['best_aic']['order'], seasonal_order=best_model_params['best_aic']['seasonal_order'], save_model=True)
