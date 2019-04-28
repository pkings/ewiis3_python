.PHONY: clean data jupyter lint requirements venv

#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_NAME = ewiis3_python_scripts
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR =  $(PROJECT_DIR)/env
JUPYTER_DIR =  $(VENV_DIR)/share/jupyter

PYTHON_INTERPRETER = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip
IPYTHON = $(VENV_DIR)/bin/ipython
JUPYTER = $(VENV_DIR)/bin/jupyter

NOTEBOOK_DIR =  $(PROJECT_DIR)/notebooks

#################################################################################
# STANDARD COMMANDS                                                             #
#################################################################################

## Install Python Dependencies
requirements: venv
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	@$(PYTHON_INTERPRETER) -m flake8 --config=$(PROJECT_DIR)/.flake8 src

# Launch jupyter server and create custom kernel if necessary
jupyter:
ifeq ($(wildcard $(JUPYTER_DIR)/kernels/$(PROJECT_NAME)/*),)
	@echo "Creating custom kernel..."
	@$(IPYTHON) kernel install --sys-prefix --name=$(PROJECT_NAME)
endif
ifeq ($(wildcard $(JUPYTER_DIR)/nbextensions/table_beautifier/*),)
	@echo "Installing jupyter notebook extensions..."
	@$(JUPYTER) contrib nbextension install --sys-prefix
	@$(JUPYTER) nbextensions_configurator enable --sys-prefix
endif
	@echo "Running jupyter notebook in background..."
	@JUPYTER_CONFIG_DIR=$(NOTEBOOK_DIR) $(JUPYTER) notebook --notebook-dir=$(NOTEBOOK_DIR)

## Install virtual environment
venv:
ifeq ($(wildcard $(VENV_DIR)/*),)
	@echo "Did not find $(VENV_DIR), creating..."
	mkdir -p $(VENV_DIR)
	python3.6 -m venv $(VENV_DIR)
endif

#################################################################################
# CUSTOM COMMANDS                                                               #
#################################################################################

## simulate prediction
simulate:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/simulation/simulate_demand_prediction.py

## grid prosumption prediction
start_grid_prosumption_predictor:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/predictors/grid_prosumption.py

## grid imbalance prediction
start_grid_imbalance_predictor:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/predictors/grid_imbalance.py

## customer prosumption prediction
start_customer_prosumption_predictor:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/predictors/customer_prosumption.py

## wholesale prices prediction
start_wholesale_prices_predictor:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/predictors/wholesale_price.py

## start model trainer
start_training_models:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/predictors/train.py

## start predicting with models
start_predicting:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/predictors/predict.py

## tariff design learner
start_tariff_design_learner:
	@$(PYTHON_INTERPRETER) src/$(PROJECT_NAME)/rl_learner/tariff_design_learner.py

## create data directories
data_dir:
	mkdir models
	mkdir logs

## cleaning data directory
clean_dirs:
	rm -r models
	rm -r logs
	make data_dir

## initial project setupt
initialize: requirements data_dir
