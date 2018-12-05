import logging
import os

import pandas as pd
from libs.time_series import TimeSeries
import customer_demand_predictor as cdp


def predict_single_customer(customer_name, time_series, forecast_length=24):
    if '{}{}.pkl'.format(customer_name, cdp.SARIMA_MODEL_SUFFIX) in os.listdir(cdp.MODEL_DIR):
        ts = TimeSeries(time_series, forecast_length, False, cdp.MODEL_DIR, cdp.SARIMA_MODEL_SUFFIX, name=customer_name)
        prediction = ts.predict_with_sarima(order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
        return prediction
    else:
        logging.info("No model has been trained yet for customer: {}.".format(customer_name))
        return None


def predict_for_all_customers(current_timeslot, df_consumption_and_production):
    customers_to_train_models = list(df_consumption_and_production['customerName'].unique())
    df_prediction = pd.DataFrame()

    for customer in customers_to_train_models:
        df_customer = df_consumption_and_production[df_consumption_and_production['customerName'] == customer]
        series_customer = df_customer['kWh']
        prediction = predict_single_customer(customer, series_customer, 24)
        for i in range(0, 24):
            df_prediction = df_prediction.append({'customer': customer, 'prediction_timeslot': current_timeslot, 'target_timeslot': current_timeslot + 1 + i, 'value': prediction[i]}, ignore_index=True)
    return df_prediction

