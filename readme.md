# OROUTER SPATIAL DATA
![Model Performance](https://i.imgur.com/s7t3QlR.png)

## Requirements
Python 3.9

## Installation

To install the necessary packages, run the following command:

```bash
pip install pandas numpy scikit-learn tensorflow

```bash
python create.py 

## Included Data 
* f1.csv - Floor 1 building, 5 minutes of 51 samples of 16k Ambient RF by ōRouter G1 running ōRouter-CLI_Miner 0.3.9
* f2.csv - Floor 2 building, 5 minutes of 51 samples, 16k ... 

# Notes
Hyperparameters tuned with keras_tuenr 64 neurons. The dense layer is a fully connected neural network layer where each input node is connected to each output node. Used dense layers in this model as its easy to read and allows the network more noise.


