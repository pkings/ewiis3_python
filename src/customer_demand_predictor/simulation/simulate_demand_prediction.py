import logging
from datetime import datetime

import pandas as pd

import customer_demand_predictor as cdp
from customer_demand_predictor import data
from customer_demand_predictor import sarima_predictor
from customer_demand_predictor import visualization

start_time_string = datetime.now().strftime('%Y-%m-%d_%H-%M')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', filename='{}{}_demand_simulation.log'.format(cdp.LOG_FILE_PATH, start_time_string),
                    level=logging.DEBUG)


def simulate(start_timeslot, end_timeslot, model_build_frequency):
    logging.info('Start simulation from timeslot {} to {}'.format(start_timeslot, end_timeslot))
    df_all_consumption_and_production = data.load_consumption_and_production_data(max_timeslot=end_timeslot+30)

    customer_filter = ['BrooksideHomes', 'CentervilleHomes', 'DowntownOffices', 'EastsideOffices', 'MedicalCenter-1']
    df_all_consumption_and_production = df_all_consumption_and_production[df_all_consumption_and_production['customerName'].isin(customer_filter)]

    if df_all_consumption_and_production.empty:
        logging.error('Failed to start simulation. Tariff transactions in db not available for selected timeslots.')

    list_prediction_dfs = []
    for current_timeslot in range(start_timeslot, end_timeslot + 1):
        logging.info('Simulate current timeslot: {}'.format(current_timeslot))
        # filter consumption array for data that is available in that timeslot
        df_known_consumption_and_production = df_all_consumption_and_production[df_all_consumption_and_production['postedTimeslotIndex'] <= current_timeslot]

        # build new models every 2 timeslots
        if current_timeslot == start_timeslot or current_timeslot % model_build_frequency == 0:
            sarima_predictor.train_models_for_each_customer(df_known_consumption_and_production)

        # predict for the next 24 timeslots
        list_prediction_dfs.append(sarima_predictor.predict_for_all_customers(current_timeslot, df_known_consumption_and_production))

    # store predictions in db
    all_predictions_df = pd.concat(list_prediction_dfs, ignore_index=True)
    data.store_prediction_to_db(all_predictions_df)


if __name__ == '__main__':
    #for modulu in range(25, 200, 25):
    modulu = 10
    start_timeslot = 500
    end_timeslot = 1600

    data.drop_prediction_table()
    simulate(start_timeslot, end_timeslot, modulu)
    visualization.boxplot_prediction_and_time_delta('prediction_performance_{}_{}_{}'.format(start_timeslot, end_timeslot, modulu))
    visualization.plot_actual_predict_for_each_customer()

