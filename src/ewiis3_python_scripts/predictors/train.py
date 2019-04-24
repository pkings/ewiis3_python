import time
from multiprocessing.dummy import Pool as ThreadPool

from ewiis3_python_scripts.predictors.customer_prosumption import CustomerProsumptionPredictor
from ewiis3_python_scripts.predictors.grid_imbalance import ImbalancePredictor
import ewiis3DatabaseConnector as data
from ewiis3_python_scripts.util import timeit


@timeit
def process_imbalance_gameId(game_id):
    try:
        imbalancePredictor = ImbalancePredictor(game_id)
        imbalancePredictor.load_data(for_training=False)

        if not imbalancePredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
        else:
            retrain_models = 20 if imbalancePredictor.get_size_of_training_data() > 40 else 2
            # training model
            if not imbalancePredictor.check_for_model_existence() or (
                        imbalancePredictor.get_size_of_training_data() % retrain_models == 0):
                imbalancePredictor.load_data(for_training=True)
                imbalancePredictor.train()
    except Exception as e:
        print("ERROR: some error has occurred during try to train imbalance model iteration.")
        print(e)


@timeit
def process_customer_prosumption_gameId(game_id):
    try:
        customerProsumptionPredictor = CustomerProsumptionPredictor(game_id)
        customerProsumptionPredictor.load_data(for_training=False)

        if not customerProsumptionPredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
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
        print("ERROR: some error has occurred during try to train customer prosumption model  iteration.")
        print(e)


def process_gameId(gameId):
    process_imbalance_gameId(gameId)
    process_customer_prosumption_gameId(gameId)


@timeit
def process_run():
    print('_______________________')
    all_gameIds = data.get_running_gameIds()
    if len(all_gameIds) == 0:
        return
    pool = ThreadPool(len(all_gameIds))
    results = pool.map(process_gameId, all_gameIds)
    pool.close()
    pool.join()


if __name__ == '__main__':
    while True:
        process_run()
        time.sleep(1.5)
