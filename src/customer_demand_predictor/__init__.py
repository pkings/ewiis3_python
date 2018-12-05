from pathlib import Path  # noqa


PROJECT_DIR = str(Path(__file__).resolve().parents[2])
MODEL_DIR = PROJECT_DIR + '/models/'
SARIMA_MODEL_SUFFIX = '_sarima_model'
LOG_FILE_PATH = PROJECT_DIR + '/logs/'
OUTPUT_PATH = PROJECT_DIR + '/output/'