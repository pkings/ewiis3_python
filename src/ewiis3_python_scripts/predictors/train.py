import time

from ewiis3_python_scripts.predictors.customer_prosumption import CustomerProsumptionPredictor
from ewiis3_python_scripts.predictors.grid_imbalance import ImbalancePredictor
import ewiis3DatabaseConnector as data


def process_imbalance_gameId(game_id):
    try:
        start_time = time.time()
        retrain_models = 20

        imbalancePredictor = ImbalancePredictor(game_id)
        imbalancePredictor.load_data(for_training=False)

        if not imbalancePredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
        else:
            # training model
            if not imbalancePredictor.check_for_model_existence() or (
                        imbalancePredictor.get_size_of_training_data() % retrain_models == 0 and imbalancePredictor.get_size_of_training_data() > 30):
                imbalancePredictor.load_data(for_training=True)
                imbalancePredictor.train()
        print('gameId: {}: grid imbalance training lasted {} seconds'.format(game_id, time.time() - start_time))
    except Exception as e:
        print("ERROR: some error has occurred during try to train imbalance model iteration.")
        print(e)


def process_customer_prosumption_gameId(game_id):
    try:
        start_time = time.time()
        retrain_models = 20

        customerProsumptionPredictor = CustomerProsumptionPredictor(game_id)
        customerProsumptionPredictor.load_data(for_training=False)

        if not customerProsumptionPredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
        else:

            # switch to saisonal model
            train_model = False
            if customerProsumptionPredictor.get_size_of_training_data() > 40 and customerProsumptionPredictor.seasonal_order is None:
                customerProsumptionPredictor.seasonal_order = (1, 0, 0, 24)
                train_model = True
            # check for (re-) training
            if not customerProsumptionPredictor.check_for_model_existence() or (
                                customerProsumptionPredictor.get_size_of_training_data() % retrain_models == 0 and customerProsumptionPredictor.get_size_of_training_data() > 30):
                train_model = True
            # train model
            if train_model:
                customerProsumptionPredictor.load_data(for_training=True)
                customerProsumptionPredictor.train()
        print('gameId: {}: customer prosumption training lasted {} seconds'.format(game_id, time.time() - start_time))
    except Exception as e:
        print("ERROR: some error has occurred during try to train customer prosumption model  iteration.")
        print(e)


def process_run():
    for game_id in data.get_running_gameIds():
        process_imbalance_gameId(game_id)
        process_customer_prosumption_gameId(game_id)


if __name__ == '__main__':
    while True:
        process_run()
        time.sleep(1.5)
