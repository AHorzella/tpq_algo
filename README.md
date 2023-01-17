# tpq_algo
***Final project for TPQ Algorithmic Trading class

Hyper-Parameter optimization of Neural-Network-Models in algorithmic trading environments: a Python-based approach

Review-Version, January 2023, Dr. Andreas Horzella


***TECHNICAL REQUIREMENTS

The project consists of following files:

(1) The orchestrating Jupyter notebook; executes the whole process step-by-step:
Optimizer_Review.jpynb

(2) Configuration file with defined parameters:
algo_config.cfg

(3) Two classes for data handling and model building:
data_environment.py, hypermodel.py

(4) Two datafiles with minutely EUR-USD price data as fixed input for the calculation:
oanda_EUR_USD_2020_01_01_2021_01_01_M1_M.csv, oanda_EUR_USD_2022_01_01_2022_10_01_M1_M.csv

For execution, several python modules are required. Beside the standard ones which are already installed on Google Colab, Keras-Tuner will additionally be installed by the main Jupyter notebook during the execution process:
%pip install keras_tuner

Furthermore, the source code will be copied from git repository to Google Colab:
!git clone https://github.com/AHorzella/tpq_algo.git

***PROJECT ABSTRACT

The increasing impact of artificial intelligence (AI) applications on more and more aspects of life is obvious. Consequently, the financial sector has also been heavily involved in corresponding developments in recent years. In terms of trading financial instruments, neural networks and reinforcement learning (RL) agents are promising approaches for building an alpha-generating algorithmic trading system: experience and gut feeling of a human trader are replaced by a complex network of calculated input-output relations. These are multi-dimensional, multi-layered and generated from existing and emerging market data using Machine Learning (ML) algorithms. The result are models that can provide statistically based forecasts of future market developments.

In addition to the availability of high-quality data, the calibration of the model parameters represents a main factor influencing the forecast quality. This includes the design of the model itself, performed by the training process, as well as the adjustment of the hyper parameters in advance, both a mutual and ongoing process in a fast-moving financial market environment.

When it comes to the design of the hyper parameters, it is the responsibility of the developer to thoroughly orchestrate the interaction of all variables. The variety of possibilities here is huge and sometimes it is quite challenging and time consuming to come up with an appropriate solution. In fact, this configuration process represents one major, time-consuming effort while building a reliable forecast model. Hence it is crucial to have supporting tools at hand which reduce the vast variety of possible configurations to the most promising ones and thereby dramatically boil down the development time for new AI models.
