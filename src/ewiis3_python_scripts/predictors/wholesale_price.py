import time
import numpy as np
import pandas as pd
import scipy.stats

from datetime import datetime
import ewiis3DatabaseConnector as db


def mean_confidence_interval(data_points, confidence=0.99):
    a = 1.0 * np.array(data_points)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



def calculate_time_delta(row):
    strFormat = "%Y-%m-%dT%H:%M:%S.%fZ"
    if row['timestamp'] is None or row['dateExecuted'] is None:
        return None

    target_timeslot = datetime.strptime(row['timestamp'], strFormat)
    order_timeslot = datetime.strptime(row['dateExecuted'], strFormat)

    time_delta = target_timeslot - order_timeslot
    return time_delta.total_seconds()/3600


def calculate_intervals_based_on_segments(segment_type):
    df_intervales = pd.DataFrame()
    for segment_type_value in list(df_cleared_trades[segment_type].unique()):
        if pd.isna(segment_type_value):
            continue
        df_cleared_trades_td = df_cleared_trades[df_cleared_trades[segment_type] == segment_type_value]
        m, lower_bound, upper_bound = mean_confidence_interval(df_cleared_trades_td['executionPrice'])
        interval = {'segment_type': segment_type, 'segment_type_value': segment_type_value, 'mean': m,
                    'lower_bound': lower_bound, 'upper_bound': upper_bound}
        df_intervales = df_intervales.append(interval, ignore_index=True)
    return df_intervales


if __name__ == '__main__':
    while True:
        try:
            start_time = time.time()
            current_game_id, latest_timeslot = db.get_current_game_id_and_timeslot()
            df_cleared_trades, game_id = db.load_cleared_trades(current_game_id)

            if df_cleared_trades.empty:
                print('No data available yet.')
            else:
                df_cleared_trades['time_delta'] = df_cleared_trades.apply(lambda row: calculate_time_delta(row), axis=1)
                df_time_delta_intervals = calculate_intervals_based_on_segments('time_delta')
                df_slotInDay_intervals = calculate_intervals_based_on_segments('slotInDay')
                confidence_intervals = pd.concat([df_time_delta_intervals, df_slotInDay_intervals])
                confidence_intervals['game_id'] = current_game_id
                db.store_price_intervals(confidence_intervals, game_id)

            print('Wholesale prices prediction (and training) lasted {} seconds'.format(time.time() - start_time))
        except Exception as e:
            print("ERROR: some error has occurred during iteration.")
            print(e)
        time.sleep(3)
