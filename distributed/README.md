# Distributed Training

# Installation and Setup

## Prerequisites
- Python 2.7
- Pip
- Virtualenv
- GCloud SDK and Google Platform Account

## Setup virtualenv
Create virtual environment:
- Mac/Linux: `virtualenv env`
- Windows: `virtualenv env`

Activate virtual environment (Must activate before running any python scripts):
- Mac/Linux: `source ./env/bin/activate`
- Windows: `.\env\Scripts\activate`

## Modules
Install the following required modules:
- `pip install --upgrade -r requirements.txt`

## Declare Config Variables
TODO

## Retrieve Dataset (Local and Cloud)
TODO

# Train Model
Activate virtual environment:
- Mac/Linux: `source ./venv/bin/activate`
- Windows: `.\venv\Scripts\activate`

## Train Locally
TODO

## Train in GCloud AI Platform (One master instance)
TODO

## Train in GCloud AI Platform - Distributed (Instances (Defined in config.yaml): 1xMaster, 6xWorkers, 3xParamServers)
TODO

Deactivate virtual environment when done by: `deactivate`