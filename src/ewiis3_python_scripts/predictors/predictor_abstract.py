from abc import ABC, abstractmethod


class PredictorAbstract(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def set_SARIMA_Parameter(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def check_for_existing_prediction(self):
        pass

    @abstractmethod
    def check_for_model_existence(self):
        pass

    @abstractmethod
    def get_size_of_training_data(self):
        pass

    @abstractmethod
    def has_enough_observations_for_training(self):
        pass
