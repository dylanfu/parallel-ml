# Parallel Machine Learning

# Installation and Setup

## Prerequisites
- Python 3.4 - 3.6 (Recommend Python 3.6.5_1)
- Pip3
- Virtualenv

## Setup virtualenv
Create virtual environment:
- Mac/Linux: `virtualenv --system-site-packages -p python3 ./venv`
- Windows: `virtualenv --system-site-packages -p python3 ./venv`

Activate virtual environment (Must activate before running any python scripts):
- Mac/Linux: `source ./venv/bin/activate`
- Windows: `.\venv\Scripts\activate`

## Modules
Install the following modules:
- `pip install six`
- `pip install wrapt`
- `pip install -U --pre tensorflow` contains the Tensorflow 2.0 Alpha release
- `pip install tfds-nightly`
- `pip install matplotlib`

For MacOS Mojave
- `pip install pyqt5`
- Add ``matplotlib.use('Qt5Agg')`` in the script or create a ~/.matplotlib/matplotlibrc file containing ``backend: Qt5Agg``

# Run scripts
Activate virtual environment:
- Mac/Linux: `source ./venv/bin/activate`
- Windows: `.\venv\Scripts\activate`

## Train Models
- To train MNIST dataset, run `python test.py`
- To train Tensorflow's Flower dataset, run `python flower.py`

Deactivate virtual environment when done by: `deactivate`

# Sequential Results 

## Torrance

### Specs
- Dell Inc. Inspiron 5567
- i7-7500U CPU @ 2.70GHz
- Turbo Boost up to 3.50GHz
- 2 Cores
- 4 Logical Processors (Hyperthreading)

### Results

#### Results for `test.py`

| Epoch     | Time (s)            | Accuracy  |
| --------- | ------------------- | --------  |
| 1         | 18.626782417297363  | 0.8822    |
| 2         | 17.893579721450806  | 0.877     |
| 3         | 17.566836833953857  | 0.8766    |
| 4         | 17.704848051071167  | 0.8725    |
| 5         | 16.981231212615967  | 0.8687    |
| Average   | 17.754655647277833  | 0.8754    |

## Dylan

### Specs
- Macbook Pro 15 2018
- i7-8850H CPU @ 2.60GHz
- Turbo Boost up to 4.30GHz
- 6 Cores
- 12 Logical Processors (Hyperthreading)

### Results

#### Results for `test.py`

| Epoch                     | Time (s)  | Accuracy  |
| ------------------------- | --------- | --------  |
| 1 (60000 Images)          | 1.92      | 0.8249    |
| 2 (60000 Images)          | 1.80      | 0.8662    |  
| 3 (60000 Images)          | 1.80      | 0.8770    |
| 4 (60000 Images)          | 1.80      | 0.8837    |
| 5 (60000 Images)          | 1.92      | 0.8914    |
| Average                   | 1.85      | 0.8686    |
| Evaluation (10000 Images) | 0.20      | 0.8728    |

#### Results for `flower.py`

| Epoch                   | Time (s)       | Training Accuracy  | Validation Accuracy  | Evaluation Accuracy  |
| ----------------------- | -------------- | ------------------ | -------------------- | -------------------- |
| 1 (2936 Images)         | 270.112        | 0.6124             | 0.6554               | -                    |
| 2 (2936 Images)         | 270.112        | 0.8975             | 0.6277               | -                    |
| 3 (2936 Images)         | 267.176        | 0.9437             | 0.6167               | -                    |
| 4 (2936 Images)         | 261.304        | 0.9576             | 0.5890               | -                    |
| 5 (2936 Images)         | 261.304        | 0.9666             | 0.6079               | -                    |
| Average                 | 266.001        | 0.8756             | 0.6193               | -                    |
| Evaluation (12 Images)  | 0.516          | -                  | -                    | 0.6611               |