import json
import os

from customer_demand_predictor import MODEL_EVALUATION_FILE_PATH, MODEL_DIR


def store_model_selection(best_models, customer):
    model_evaluations = load_model_selection()
    model_evaluations[customer] = best_models
    with open(MODEL_EVALUATION_FILE_PATH, 'w') as fp:
        json.dump(model_evaluations, fp)


def load_model_selection():
    approach_calculations = {}
    if os.path.isfile(MODEL_EVALUATION_FILE_PATH):
        approach_calculations = json.load(open(MODEL_EVALUATION_FILE_PATH))
    return approach_calculations


def check_for_model_existence(model_path):
    return os.path.isfile(model_path)


def build_model_save_path(target, type, model_name):
    return '{}{}_{}_{}.pkl'.format(MODEL_DIR, target, type, model_name)
