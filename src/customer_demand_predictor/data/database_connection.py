import pymysql
import pandas as pd
import logging
import time
from sqlalchemy import create_engine

db_user = 'root'
db_pw = 'Sorge6,aastet'
db_host = 'localhost'
db_port = 3306
db_schema = 'ewiis3'


def __connect_to_local_database():
    conn = pymysql.connect(host=db_host, user=db_user, passwd=db_pw, db=db_schema)
    return conn


def create_db_connection_engine():
    connection_string = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(db_user, db_pw, db_host, db_port, db_schema)
    cnx = create_engine(connection_string, echo=False)
    return cnx


def execute_sql_query(sql_query):
    conn = __connect_to_local_database()
    df_mysql = pd.read_sql(sql_query, con=conn)
    conn.close()
    return df_mysql


def load_consumption_and_production_data(max_timeslot=None):
    where_clause = '' if max_timeslot == None else ' WHERE `postedTimeslotIndex` <= {}'.format(max_timeslot)
    sql_statement = "SELECT * FROM `tariff_transaktion`{}".format(where_clause)
    df_tariff_transactions = execute_sql_query(sql_statement)
    filter_tx_type = ['CONSUME', 'PRODUCE']
    df_tariff_transactions = df_tariff_transactions[df_tariff_transactions['txType'].isin(filter_tx_type)]
    return df_tariff_transactions


def load_cleared_trades(game_id):
    if game_id is None:
        return pd.DataFrame(), game_id

    try:
        sql_statement = 'SELECT ct.*, ts.isWeekend, ts.dayOfWeek, ts.slotInDay, ts.timestamp FROM (SELECT t.* FROM ewiis3.cleared_trade t WHERE gameId="{}") AS ct LEFT JOIN (SELECT * FROM ewiis3.timeslot WHERE timeslot.gameId="{}") AS ts ON ct.timeslot = ts.serialNumber'.format(game_id, game_id)
        df_cleared_trades = execute_sql_query(sql_statement)
    except Exception as e:
        print('Error occured while requesting cleared trades from db.')
        df_cleared_trades = pd.DataFrame()
    return df_cleared_trades, game_id


def load_predictions(table_name):
    try:
        sql_statement = "SELECT * FROM `{}`".format(table_name)
        df_predictions = execute_sql_query(sql_statement)
    except Exception as e:
        print('Error occured while requesting `{}` table from db.'.format(table_name))
        df_predictions = pd.DataFrame()
    return df_predictions


def load_total_grid_consumption_and_production(game_id):
    if game_id is None:
        return pd.DataFrame(), game_id

    start_time = time.time()
    try:
        sql_statement = 'SELECT prosumptin_meets_weather.*, ts.isWeekend, ts.dayOfWeek, ts.slotInDay FROM (SELECT dr.*, wr.cloudCover, wr.temperature, wr.windDirection, wr.windSpeed FROM (SELECT * FROM ewiis3.distribution_report WHERE distribution_report.gameId="{}") AS dr LEFT JOIN (SELECT * FROM ewiis3.weather_report WHERE weather_report.gameId="{}") AS wr ON dr.timeslot = wr.timeslotIndex) AS prosumptin_meets_weather LEFT JOIN (SELECT * FROM ewiis3.timeslot WHERE timeslot.gameId="{}") AS ts ON prosumptin_meets_weather.timeslot = ts.serialNumber;'.format(game_id, game_id, game_id)
        df_total_grid_consumption_and_production = execute_sql_query(sql_statement)
    except Exception as e:
        print('Error occured while requesting grid consumption and production from db.')
        df_total_grid_consumption_and_production = pd.DataFrame()
    print('Loading grid consumption and production last: {} seconds.'.format(time.time() - start_time))
    return df_total_grid_consumption_and_production, game_id


def store_predictions(df_prosumption_predictions, table_name):
    cnx = create_db_connection_engine()
    df_prosumption_predictions.to_sql(name=table_name, schema='ewiis3', con=cnx, if_exists='append', index=False)


def load_grid_imbalance(game_id):
    if game_id is None:
        return pd.DataFrame(), game_id

    start_time = time.time()
    try:
        sql_statement = 'SELECT prosumptin_meets_weather.*, ts.isWeekend, ts.dayOfWeek, ts.slotInDay FROM (SELECT dr.*, wr.cloudCover, wr.temperature, wr.windDirection, wr.windSpeed FROM (SELECT * FROM ewiis3.balance_report WHERE balance_report.gameId="{}") AS dr LEFT JOIN (SELECT * FROM ewiis3.weather_report WHERE weather_report.gameId="{}") AS wr ON dr.timeslotIndex = wr.timeslotIndex) AS prosumptin_meets_weather LEFT JOIN (SELECT * FROM ewiis3.timeslot WHERE timeslot.gameId="{}") AS ts ON prosumptin_meets_weather.timeslotIndex = ts.serialNumber;'.format(game_id, game_id, game_id)
        df_total_grid_imbalance = execute_sql_query(sql_statement)
    except Exception as e:
        print('Error occured while requesting grid imbalances from db.')
        df_total_grid_imbalance = pd.DataFrame()
    print('Loading grid imbalance last: {} seconds.'.format(time.time() - start_time))
    return df_total_grid_imbalance, game_id


def load_customer_prosumption(game_id):
    if game_id is None:
        return pd.DataFrame(), game_id

    start_time = time.time()
    try:
        sql_statement = 'SELECT * FROM (SELECT * FROM (SELECT postedTimeslotIndex, SUM(kWH) FROM ewiis3.tariff_transaktion WHERE gameId = "{}" AND (txType = "CONSUME" OR txType = "PRODUCE") GROUP BY postedTimeslotIndex) AS customer_prod_con LEFT JOIN (SELECT * FROM ewiis3.weather_report WHERE weather_report.gameId="{}") AS wr ON customer_prod_con.postedTimeslotIndex = wr.timeslotIndex ) AS prosumption_meets_weather LEFT JOIN (SELECT * FROM ewiis3.timeslot WHERE gameId = "{}") AS ts ON prosumption_meets_weather.postedTimeslotIndex = ts.serialNumber;'.format(game_id, game_id, game_id)
        df_customer_prosumption = execute_sql_query(sql_statement)
    except Exception as e:
        print('Error occured while requesting customer prosumption from db.')
        df_customer_prosumption = pd.DataFrame()
    print('Loading customer prosumption last: {} seconds.'.format(time.time() - start_time))
    return df_customer_prosumption, game_id


def get_current_game_id_and_timeslot():
    game_id = None
    latest_timeslot = None
    try:
        sql_statement='SELECT * FROM ewiis3.timeslot ORDER BY timeslotId DESC LIMIT 1;'
        df_latest = execute_sql_query(sql_statement)
        latest_timeslot = df_latest['serialNumber'].values[0]
        game_id = df_latest['gameId'].values[0]
    except Exception as e:
        print('Error occured while requesting current game_id and latest timeslot from db.')
    return game_id, latest_timeslot


def store_price_intervals(df_intervals):
    try:
        conn = __connect_to_local_database()
        conn.cursor()
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE ewiis3.wholesale_price_intervals")
        conn.commit()
        conn.close()

        cnx = create_db_connection_engine()
        df_intervals.to_sql(name='wholesale_price_intervals', schema='ewiis3', con=cnx, if_exists='append',
                                          index=False)
    except Exception as e:
        print('Error occured during storing price intervals.')
        print(e)
