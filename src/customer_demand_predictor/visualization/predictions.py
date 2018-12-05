import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import customer_demand_predictor as cdp
from customer_demand_predictor import data


def boxplot_prediction_and_time_delta(filename):
    df_consumption_and_production = data.load_consumption_and_production_data()
    df_consumption_and_production.rename(columns={'postedTimeslotIndex': 'target_timeslot', 'customerName': 'customer'}, inplace=True)
    df_predictions = data.load_predictions()

    result = pd.merge(df_predictions, df_consumption_and_production, how='left', on=['target_timeslot', 'customer'])
    result['target_timeslot'] = pd.to_numeric(result['target_timeslot'])
    result['time_delta'] = result['target_timeslot'] - result['prediction_timeslot']
    result.dropna(subset=['kWh'], inplace=True)  # drop predictions, that exceed the actual values
    result['mae'] = abs(result['kWh']-result['value'])
    df_aggregated_consumption = result[['target_timeslot', 'prediction_timeslot', 'time_delta', 'value', 'kWh']].groupby(by=['target_timeslot', 'prediction_timeslot', 'time_delta'], as_index=False).sum()
    df_aggregated_consumption['mae'] = abs(df_aggregated_consumption['kWh']-df_aggregated_consumption['value'])

    fig = plt.figure(figsize=(15, 20))
    ax1 = fig.add_subplot(211)
    plt.title('Prediction performance', fontsize=12)
    g = sns.boxplot(ax=ax1, x="time_delta", y="mae", hue='customer', data=result)
    ax2 = fig.add_subplot(212)
    g2 = sns.boxplot(ax=ax2, x="time_delta", y="mae", data=df_aggregated_consumption)
    fig.tight_layout()
    plt.savefig('{}{}'.format(cdp.OUTPUT_PATH, filename))
    logging.info('Successfully created boxplot for prediction and time delta.')

def plot_actual_predict_for_each_customer():
    df_consumption_and_production = data.load_consumption_and_production_data()
    df_consumption_and_production.rename(columns={'postedTimeslotIndex': 'target_timeslot', 'customerName': 'customer'},
                                         inplace=True)
    df_predictions = data.load_predictions()

    result = pd.merge(df_predictions, df_consumption_and_production, how='left', on=['target_timeslot', 'customer'])
    result['target_timeslot'] = pd.to_numeric(result['target_timeslot'])
    result['time_delta'] = result['target_timeslot'] - result['prediction_timeslot']
    result.dropna(subset=['kWh'], inplace=True)  # drop predictions, that exceed the actual values

    customers = list(result['customer'].unique())
    for customer in customers:
        df_customer = result[result['customer'] == customer]
        if df_customer.empty:
            print('no predictions for customer {}.'.format(customer))
            continue
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(111)
        plt.title('Prediction performance for {}'.format(customer), fontsize=12)
        # g = sns.lineplot(ax=ax1, x="target_timeslot", y="value", hue='time_delta', data=df_customer, palette=sns.color_palette("Blues_d", n_colors=24))
        # g = sns.lineplot(ax=ax1, x="target_timeslot", y="kWh", data=df_customer, label='actual')
        g = sns.lineplot(ax=ax1, x="target_timeslot", y="value", data=df_customer, label='prediction')
        g = sns.lineplot(ax=ax1, x="target_timeslot", y="kWh", data=df_customer, label='actual')
        fig.tight_layout()
        plt.savefig('{}{}_actual_vs_predict'.format(cdp.OUTPUT_PATH, customer))
        logging.info('Successfully created actual vs. prediction plot for {}'.format(customer))

if __name__ == '__main__':
    plot_actual_predict_for_each_customer()
