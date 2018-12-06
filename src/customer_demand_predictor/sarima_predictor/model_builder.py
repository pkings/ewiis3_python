import customer_demand_predictor as cdp
from customer_demand_predictor.sarima.time_series import TimeSeries
import pandas as pd


def train_models_for_each_customer(df_consumption_and_production):
    customers_to_train_models = list(df_consumption_and_production['customerName'].unique())

    for customer in customers_to_train_models:
        df_customer = df_consumption_and_production[df_consumption_and_production['customerName'] == customer]
        df_customer = df_customer.set_index(pd.DatetimeIndex(start='2018-01-01 00:00:00', freq='H', periods=len(df_customer.index)))  # delme: just for x13 purpose
        series_customer = df_customer['kWh'][-96:]  # train on last 96 timesots
        print(series_customer.index)  # delme: just for x13 purpose
        time_series = TimeSeries(series_customer, 24, False, cdp.MODEL_DIR, cdp.SARIMA_MODEL_SUFFIX, name=customer)
        best_orders = time_series.find_best_orders()  # delme: just for x13 purpose
        print(best_orders)  # delme: just for x13 purpose
        time_series.train_sarima_model(order=(1, 0, 0), seasonal_order=(1, 0, 0, 24), save_model=True)
