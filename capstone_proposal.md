---
title: "capstone proposal"
author: "Tu Ha"
date: "May 29, 2018"
output: html_document
---

# Machine Learning Engineer Nanodegree
## Capstone Proposal
Tu Ha

May 29th, 2018

## Proposal

# Apply Error-Trend-Seasonal model for time series forecasting

### Domain Background

Time series forecasting is required in a lot of domains as weather forecast, earthquake prediction, finance forecast, or population forecast. Forecasting tasks help us to foresee disasters, risks or chances emerging from nature, social, financial system, etc. A plan or action is then established to overcome or reduce damages or take advantage of the chances. Researchers have been doing a number of researches to find methodologies and techniques to understand the time series data, and the way to predict future events. There are two methodologies to build up a model for forecasting, the global modeling and local modeling. The global modeling is useful in figuring out the underlying structures/patterns in the dataset, while the local modeling adapts better to the dynamics of data. Data of stock trading is pretty high dynamic. The attention of local modeling should be paid in this case.

The local modeling methodology takes into account the nearest neighbour data points around a query point to build up a local model which is expected to explain this data group as much as possible. The local model is used to perform the forecasting task with an assumption that the future outcome is similar to the behavior of the local model. Two well-known modeling methods are the maximum likelihood (ML) estimation and the maximum a posteriori (MAP) estimation. The Takens' theorem is applied in the ML estimation method [1] to predict future events. However finding a plausible window of local data requires a lot of calculation. The MAP estimation method applies the Bayes rule to a prediction-correction framework, which needs state transition to be defined clearly. Error-Trend-Seasonal (ETS) model [2] which implements the MAP estimation method is used for  the time series forecasting in this project.





### Problem Statement

Local modeling methodology rebuids a new local model for each prediction period. Therefore estimating model parameters become critical problem. If the model is so simple, parameters estimation is fast but the model may not capture enough behaviors of the local data. On the other hand, so many parameters cost a lot of time to estimate. The size of the local data is another problem. How to identify the neighborhood size to capture enough behaviors for the prediction. Interpreting forecasting result becomes easier if a model is able to provide confidence intervals around the point forecasting. These three problems inspire researchers to propose better models.

The combination of Takens' theorem and the ML estimation method [1] defines a local modeling method which provides point forecasting only. This local modeling tries to predict all kinds of dynamics of the system by looking for similar dynamics in the past. So searching for highly-correlated patterns in the local data to predict the current pattern requires a lot of calculation.

The statiscal theory behide the ETS model helps to obtain confidence intervals together with the point forecasting. The model classifies the dynamics of the system into several simple ones, and tries to model each dynamics component. These components are then combined to build up the final model. Decomposing the dynamics into simpler single ones helps to reduce the calculation time, and high performance could be archieved if the combination matches the actual dynamics.

Choosing proper size of local data needs heuristics. [1] proposed an exhaustic search to find the optimal size, but the searching range is still specified by users.


### Datasets and Inputs

The historical data of trading Apple stock (AAPL) in three years from the beginning of 2015 to the end of 2017 is obtained from the Quandl platform, and used as the dataset for this project. This is the time series data with pretty high dynamics, and finding a proper ETS model for this dataset is a challenge. 

Investors and managers often concern the return rather than the price. So the input to the model is the return, and the output is the predicted daily return of AAPL. 

### Solution Statement

The dataset is divided into the training and test dataset. The data in the year 2015 and 2016 is the training set and the year 2017 is the test set.The dynamics of the training set is analyzed using the Alteryx tool. The analyzing result helps to select several plausible ETS models and the data windows for each model. The model which has the best performance on the training set is chosen to perform the forecast on the test set. The metric to measure performance is described in the below part.

A fixed-size window is allocated for each ETS model. Model parameters and sigma of the noise model are estimated within this window. One-step predictions are then generated by each model. The window moves one step forward, and the parameters and sigma are estimated again. The next one-step predictions are generated. This process repeats on all ETS models on the training dataset to obtain the performance curves. The best-performance model is selected to make predictions on the test dataset.


### Benchmark Model

Several simple prediction models could be used as the benchmark model.
* Naive prediction model which generates one-step prediction equal to the current value.
* Simple extrapolation model which extends the current segment connecting the current value with the previous one.
* Random walk model in which the mean is the current value, and the sigma equals to the estimated sigma of the error component in the ETS model.

All these models are used as the tree benchmark models in this project.

### Evaluation Metrics

The following metrics [3] are used to evaluate how precise the prediction is.
* The root mean squared error (RMSE)
* The sum of squared error (SSE)
* The mean absolute error (MAE)
* The mean absolute percentage error (MAPE)
* The mean absolute scaled error (MASE)

### Project Design

The methodology to realize the project is the bottom-up approach. The project could be broken down to the following modules.

* Data analysis by the Alteryx tool to figure out characteristics of basic dynamics Trending, Season, and Error.
* Select several ETS models basing on the above analysis.
* Derive the optimizing formula [3] for each model.
* Derive the five metric formula [3] to evaluate performance for each model.
* Estimate the model parameters and the size of the data window for each model to archieve the best performance.
* Visualize the predicting curves and performance curves of all ETS models.
* Choose the model with the best performace and run the model with the test dataset.
* Feed the test dataset to the random walk model to obtain the benchmark data.
* Visualize the predicting curves and performance curves of the best model and the random walk model.
* Give conclusions about the ETS model applied to forecast the daily return of AAPL stock.

-----------
[1] James McNames, Innovations in local modeling for time series prediction, PhD. dissertation, 1999.

[2] https://www.otexts.org/fpp/7

[3] Rob J. Hyndman, J. K Ord, Anne B. Koehler, Ralph D. Snyder, Forecasting with exponential smoothing: The state space approach, 2008, Springer.

