import numpy as np

class performance_measurement():
    def __init__(self):
        # performance metrics
        self.n    = 0  # the number of observations
        self.sse  = 0  # sum of squared errors to measure performance
        self.sse_s= 0  # sum of squared errors to estimate sigma
        self.rmse = 0  ## 
        self.mae  = 0  # mean absolute error
        self.mse  = 0  # mean squared error
        self.mape = 0  # mean absolute percentage error
        self.smape= 0  # symmetric mean absolute percentage error
        self.mase = 0  ## mean absolute scaled error

    def error_metrics(self, y_observed, y_hat):
        '''
        y_observed: the observed value at the period t=0
        y_hat: the estimated value at the same period as y_observed
        '''
        error = abs(y_observed - y_hat)
        #print("y_observed={} y_hat={} error={}".format(y_observed, y_hat, error))
        symmetric_error =  error / np.maximum(y_observed + y_hat, 0.01) * 200
        percentage = error / np.maximum(y_observed * 100, 0.01)

        self.sse   += error**2
        self.mae    = (self.mae   * (self.n-1) + error) / self.n
        self.mse    = (self.mse   * (self.n-1) + (error**2)) / self.n
        self.mape   = (self.mape  * (self.n-1) + np.abs(percentage)) / self.n
        self.smape  = (self.smape * (self.n-1) + symmetric_error) / self.n

    def update_sigma2(self, y_observed, y_hat):
        '''
        y_observed: the observed value at the period t=0
        y_hat: the estimated value at the same period as y_observed
        '''
        error = np.maximum(abs(y_observed - y_hat), 0.0001) # avoid the initial error = 0
        #print("y_observed={} y_hat={} error={}".format(y_observed, y_hat, error))

        self.sse_s += error**2             # sum of squared errors
        self.sigma2 = self.sse_s / self.n  # estimated squared sigma to calculate prediction intervals


class ETS(performance_measurement):
    def __init__(self, l0=0, b0=0, s0=0, alpha=0, beta=0, gamma=0, m=1, phi=0):
        '''
        l0 : the level value at period t=0
        b0 : the trend value at period t=0
        s0 : the season value at period t=0
        m : the period of the seasonality, m=4 for quarterly, 12 for monthly, must be >= 1.
        '''
        performance_measurement.__init__(self)

        # parameters
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.m     = m
        self.phi   = phi

        # initial state at the period t=0
        self.l = l0   
        self.b = b0
        #print("m=",m)
        self.s = np.ones(m) * s0 # [s_t, s_t-1, s_t-2, ..., s_t-m+1]

        # prediction variance simulation
        self.sample_size = 1000

    def predict_observe(self, y_observed):
        '''
        y_observed: the observation at the period t=1.

        Return the predicted value y_hat and its confident interval for the next period t=1,
        the order of the following steps is important.

        Calculate also the performance of the prediction process.
        '''
        self.n += 1

        # 1.Predict y_hat for the next period t=1
        y_hat = self.predict()

        # 2.Update sigma to estimate prediction interval later.
        self.update_sigma2(y_observed, y_hat)

        # 3.Calculate several error metrics of the current prediction
        self.error_metrics(y_observed, y_hat)

        # 4.Update internal state at period t=0 with the observation at period t=1
        self.observe(y_observed)
        # at this moment, the state of the model is at the period t=1
        #                 shift the time window => the state is at t=0 again and ready for the next prediction.

        # 5.Get confident interval
        y_lower = self.pred_interval[0]
        y_upper = self.pred_interval[1]

        return y_hat, y_lower, y_upper


    def observe(self, y_observed):
        # Update internal states to the same period of the measurement at t=1,
        # the update order is important
        self.error(y_observed)
        self.level()
        self.trend()
        self.season()
        self.update()

