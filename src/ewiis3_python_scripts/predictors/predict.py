import time

from ewiis3_python_scripts.predictors.customer_prosumption import CustomerProsumptionPredictor
from ewiis3_python_scripts.predictors.grid_imbalance import ImbalancePredictor
import ewiis3DatabaseConnector as data


def process_imbalance_gameId(game_id):
    try:
        start_time = time.time()

        imbalancePredictor = ImbalancePredictor(game_id)
        imbalancePredictor.load_data(for_training=False)

        if not imbalancePredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
        else:
            # predict
            if imbalancePredictor.check_for_model_existence() and not imbalancePredictor.check_for_existing_prediction():
                imbalancePredictor.predict()
        print('gameId: {}: grid imbalance prediction lasted {} seconds'.format(game_id, time.time() - start_time))
    except Exception as e:
        print("ERROR: some error has occurred during try to predict imbalance iteration.")
        print(e)


def process_customer_prosumption_gameId(game_id):
    try:
        start_time = time.time()

        customerProsumptionPredictor = CustomerProsumptionPredictor(game_id)
        customerProsumptionPredictor.load_data(for_training=False)

        if not customerProsumptionPredictor.has_enough_observations_for_training():
            print('Not enough data to build models and predict')
        else:

            # switch to saisonal model
            if customerProsumptionPredictor.get_size_of_training_data() > 40 and customerProsumptionPredictor.seasonal_order is None:
                customerProsumptionPredictor.seasonal_order = (1, 0, 0, 24)
            # predict
            if customerProsumptionPredictor.check_for_model_existence() and not customerProsumptionPredictor.check_for_existing_prediction():
                customerProsumptionPredictor.predict()
        print('gameId: {}: customer prosumption prediction lasted {} seconds'.format(game_id, time.time() - start_time))
    except Exception as e:
        print("ERROR: some error has occurred during try to predict customer prosumption iteration.")
        print(e)


def process_run():
    for game_id in data.get_running_gameIds():
        process_imbalance_gameId(game_id)
        process_customer_prosumption_gameId(game_id)


if __name__ == '__main__':
    while True:
        process_run()
        time.sleep(1.5)
