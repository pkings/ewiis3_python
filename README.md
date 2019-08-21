# EWIIS3 Python Scripts
This repository contains Python scripts of the EWIIS3 broker for predicting electricity demand and the imbalance of the grid using SARIMAX time series models.

## Installation
1. run `make initialize`

### database setup
configure the values in the .env file in the main directory
```
DB_USER=<user>
DB_HOST=127.0.0.1
DB_PW=<password>
DB_PORT=3306
DB_SCHEMA=ewiis3
```
## start training and predicting
1. run `make start_training_models`
2. run `make start_predicting`
