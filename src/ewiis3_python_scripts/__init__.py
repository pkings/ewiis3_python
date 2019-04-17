from pathlib import Path  # noqa

from gym.envs.registration import register

PROJECT_DIR = str(Path(__file__).resolve().parents[2])
MODEL_DIR = PROJECT_DIR + '/models/'
MODEL_EVALUATION_FILE_PATH = MODEL_DIR + 'model_selection.json'
SARIMA_MODEL_SUFFIX = '_sarima_model'
LOG_FILE_PATH = PROJECT_DIR + '/logs/'
OUTPUT_PATH = PROJECT_DIR + '/output/'

register(id="tariff-v0", entry_point="customer_demand_predictor.envs:TariffEnv")
