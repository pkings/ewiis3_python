import pymysql
import pandas as pd
import logging
from sqlalchemy import create_engine

db_user = 'root'
db_pw = 'Web2)wanted'
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


def load_predictions():
    sql_statement = "SELECT * FROM `prediction`"
    df_predictions = execute_sql_query(sql_statement)
    return df_predictions


def drop_prediction_table():
    conn = __connect_to_local_database()
    conn.cursor()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS prediction")
    conn.commit()
    conn.close()
    logging.info('Successfully dropped prediction table in local db.')


def store_prediction_to_db(df_predictions):
    cnx = create_db_connection_engine()
    df_predictions.to_sql(name='prediction', schema='ewiis3', con=cnx, if_exists='append', index=False)
    logging.info('Successfully stored all predictions in db.')
