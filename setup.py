from setuptools import find_packages, setup

setup(
    name='ewiis3_python_scripts',
    packages=find_packages('src'),
    version='0.0.1',
    description='Analyse and predict for ewiis3 broker',
    author='Peter Kings',
    package_dir={"": "src"},
)
