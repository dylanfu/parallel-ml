# parallel-ml

# Installation

## Prerequisites
- Python 3.4 +
- Pip3
- Virtualenv

## Modules
Install the following modules:
- `pip install tensorflow`
- `pip install matplotlib`

For MacOS Mojave
- `pip install pyqt5`
- Add ``matplotlib.use('Qt5Agg')`` in the script or create a ~/.matplotlib/matplotlibrc file containing ``backend: Qt5Agg``

# Run the training script
Run `python test.py`

# Sequential Results (Torrance)

## Specs
- Dell Inc. Inspiron 5567
- i7-7500U CPU @ 2.70GHz
- 2901 Mhz
- 2 Cores
- 4 Logical Processors (Hyperthreading)

## Results

Iteration | Time (s)            | Accuracy
-------------------------------------------
1         | 18.626782417297363  | 0.8822
2         | 17.893579721450806  | 0.877
3         | 17.566836833953857  | 0.8766
4         | 17.704848051071167  | 0.8725
5         | 16.981231212615967  | 0.8687
Average   | 17.754655647277833  | 0.8754
