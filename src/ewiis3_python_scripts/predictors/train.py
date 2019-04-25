import time
from multiprocessing.dummy import Pool as ThreadPool
import logging
from datetime import datetime

from ewiis3_python_scripts.predictors.customer_prosumption import CustomerProsumptionPredictor
from ewiis3_python_scripts.predictors.grid_imbalance import ImbalancePredictor
import ewiis3DatabaseConnector as data
from ewiis3_python_scripts.util import timeit

start_time_string = datetime.now().strftime('%Y-%m-%d_%H-%M')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', filename='logs/' + start_time_string + '_trainer.log',
                    level=logging.DEBUG)



@timeit
def process_imbalance_gameId(game_id):
    try:
        imbalancePredictor = ImbalancePredictor(game_id)
        imbalancePredictor.load_data(for_training=False)

        if not imbalancePredictor.has_enough_observations_for_training():
            logging.info('Not enough data to build models and predict')
        else:
            retrain_models = 20 if imbalancePredictor.get_size_of_training_data() > 40 else 2
            # training model
            if not imbalancePredictor.check_for_model_existence() or (
                        imbalancePredictor.get_size_of_training_data() % retrain_models == 0):
                imbalancePredictor.load_data(for_training=True)
                imbalancePredictor.train()
    except Exception as e:
        logging.error("ERROR: some error has occurred during try to train imbalance model iteration.")
        logging.error(e)


@timeit
def process_customer_prosumption_gameId(game_id):
    try:
        customerProsumptionPredictor = CustomerProsumptionPredictor(game_id)
        customerProsumptionPredictor.load_data(for_training=False)

        if not customerProsumptionPredictor.has_enough_observations_for_training():
            logging.info('Not enough data to build models and predict')
        else:

            # switch to saisonal model
            train_model = False
            if customerProsumptionPredictor.get_size_of_training_data() > 40 and customerProsumptionPredictor.seasonal_order is None:
                customerProsumptionPredictor.seasonal_order = (1, 0, 0, 24)
            if customerProsumptionPredictor.get_size_of_training_data() > 40 and customerProsumptionPredictor.get_size_of_training_data() < 50:  # to be sure to train for seasonal
                train_model = True

            retrain_models = 20 if customerProsumptionPredictor.get_size_of_training_data() > 40 else 2
            # check for (re-) training
            if not customerProsumptionPredictor.check_for_model_existence() or (
                                customerProsumptionPredictor.get_size_of_training_data() % retrain_models == 0):
                train_model = True
            # train model
            if train_model:
                customerProsumptionPredictor.load_data(for_training=True)
                customerProsumptionPredictor.train()
    except Exception as e:
        logging.error("ERROR: some error has occurred during try to train customer prosumption model  iteration.")
        logging.error(e)


def process_gameId(gameId_and_pred_type):
    if gameId_and_pred_type['type'] == 'imba':
        process_imbalance_gameId(gameId_and_pred_type['gameId'])
    elif gameId_and_pred_type['type'] == 'cusPro':
        process_customer_prosumption_gameId(gameId_and_pred_type['gameId'])


@timeit
def process_run():
    logging.info('_______________________')
    all_gameIds = data.get_running_gameIds()
    if len(all_gameIds) == 0:
        return
    process_gameId_and_pred = []
    for gameId in all_gameIds:
        process_gameId_and_pred.append({'gameId': gameId, 'type': 'imba'})
        process_gameId_and_pred.append({'gameId': gameId, 'type': 'cusPro'})

    pool = ThreadPool(8)
    results = pool.map(process_gameId, process_gameId_and_pred)
    pool.close()
    pool.join()


if __name__ == '__main__':
    while True:
        process_run()
        time.sleep(1.5)
