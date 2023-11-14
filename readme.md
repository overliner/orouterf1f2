# OROUTER SPATIAL ANALYSES
![Model Performance](https://i.imgur.com/s7t3QlR.png)

## Intro
The ōRouter is capable of capturing and analyzing ELF ambient radio frequency (RF) signals. It operates by creating RF "fingerprints" of static and extracting features that are unique to the location. This unique approach allows for smart spatial applications. 

With its precise location-based RF analysis, the ōRouter can be utilized to visualize and interact with metaverse assets anchored to specific real-world locations. This functionality enables a seamless blend of digital and physical worlds, enhancing user experiences in augmented reality (AR) and virtual reality (VR) applications. In the domain of transportation and shipping, the ōRouter aids in managing the delivery and execution of transactions. Its ability to provide detailed spatial data ensures efficient route planning, real-time tracking, and enhanced security for transported goods. This is especially useful for companies seeking to optimize their supply chain and logistics operations through technology.

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


