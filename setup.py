from setuptools import find_packages, setup

setup(
    name='customer_demand_predictor',
    packages=find_packages('src'),
    version='0.0.1',
    description='Analyse and predict the demand of a brokers customer',
    author='Peter Kings',
    package_dir={"": "src"},
)
