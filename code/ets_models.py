# implement innovations state space model
# 30 ETS models
# https://www.otexts.org/sites/default/files/fpp/images/Table7-10.png

import numpy as np
from scipy.stats import norm
from ets import ETS

###################################
### Additive Error Models       ###
###################################
# 1.ETS(A,N,N) - class 1 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_ANN(ETS):
    def __str__(self):
        return ("{'l0':%s,'alpha':%s,'conf_interval':%s}" %
                (self.l,self.alpha,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, alpha=a)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + alpha * e_t+1
        self.l1 = self.l + self.alpha * self.e1

    def trend(self):
        pass

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        # update the prediction interval
        var = self.sigma2
        #self.pred_interval.append(norm.interval(alpha=self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(alpha=self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t
        self.y_hat = self.l
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1



# 2.ETS(A,N,A) - class 1 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_ANA(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'s0':%s,'alpha':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,ss,self.alpha,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, s0=s0, alpha=a, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + alpha * e_t+1
        self.l1 = self.l + self.alpha * self.e1

    def trend(self):
        pass

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * e_t+1
        self.s1 = self.s[self.m-1] + self.gamma * self.e1

    def update(self):
        self.l = self.l1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s to prepare for prediction
        # update the prediction interval
        var = self.sigma2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase


    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + s_t-m+1
        self.y_hat = self.l + self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 3.ETS(A,N,M) - class 5 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_ANM(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'s0':%s,'alpha':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,ss,self.alpha,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, s0=s0, alpha=a, gamma=g, m=m)
 
        # object variable
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + alpha * e_t+1 / s_t-m+1
        self.l1 = self.l + self.alpha * self.e1 / np.maximum(self.s[self.m-1], 0.01)

    def trend(self):
        pass

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * e_t+1 / l_t
        self.s1 = self.s[self.m-1] + self.gamma * self.e1 / np.maximum(self.l, 0.01)

    def update(self):
        self.l = self.l1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high]))
        self.pred_interval = (self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high])
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * s_t-m+1
        self.y_hat = self.l * self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 4.ETS(A,A,N) - class 1 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AAN(ETS):
    def __str__(self):
        return ("{'l0':%s,'b0':%s,'alpha':%s,'beta':%s,'conf_interval':%s}" %
                (self.l,self.b,self.alpha,self.beta,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, alpha=a, beta=b)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + b_t + alpha * e_t+1
        self.l1 = self.l + self.b + self.alpha * self.e1

    def trend(self):
        # b_t+1 = b_t + beta * e_t+1
        self.b1 = self.b + self.beta * self.e1

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        self.b = self.b1
        # update the prediction interval
        var = self.sigma2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + b_t
        self.y_hat = self.l + self.b
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 5.ETS(A,A,A) - class 1 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AAA(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + b_t + alpha * e_t+1
        self.l1 = self.l + self.b + self.alpha * self.e1

    def trend(self):
        # b_t+1 = b_t + beta * e_t+1
        self.b1 = self.b + self.beta * self.e1
        pass

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * e_t+1
        self.s1 = self.s[self.m-1] + self.gamma * self.e1

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + b_t + s_t-m+1
        self.y_hat = self.l + self.b + self.s[self.m-1]

        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 6.ETS(A,A,M) - class 5 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AAM(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + b_t + alpha * e_t+1 / s_t-m+1
        self.l1 = self.l + self.b + self.alpha * self.e1 / np.maximum(self.s[self.m-1], 0.01)

    def trend(self):
        # b_t+1 = b_t + beta * e_t+1 / (s_t-m+1)
        self.b1 = self.b + self.beta * self.e1 / np.maximum(self.s[self.m-1], 0.01)

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * e_t+1 / (l_t + b_t)
        self.s1 = self.s[self.m-1] + self.gamma * self.e1 / np.maximum(self.l + self.b, 0.01)

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high]))
        self.pred_interval = (self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high])
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = (l_t + b_t) * s_t-m+1
        self.y_hat = (self.l + self.b) * self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 7.ETS(A,Ad,N) - class 1 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AAdN(ETS):
    def __str__(self):
        return ("{'l0':%s,'b0':%s,'alpha':%s,'beta':%s,'phi':%s,'conf_interval':%s}" %
                (self.l,self.b,self.alpha,self.beta,self.phi,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, alpha=a, beta=b, phi=p)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t + phi * b_t + alpha * e_t+1
        self.l1 = self.l + self.phi * self.b + self.alpha * self.e1

    def trend(self):
        # b_t+1 = phi * b_t + beta * e_t+1
        self.b1 = self.phi * self.b + self.beta * self.e1

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        self.b = self.b1
        # update the prediction interval
        var = self.sigma2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + phi * b_t
        self.y_hat = self.l + self.phi * self.b
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 8.ETS(A,Ad,A)
# 9.ETS(A,Ad,M)

# 10.ETS(A,M,N) - class 5 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AMN(ETS):
    def __str__(self):
        return ("{'l0':%s,'b0':%s,'alpha':%s,'beta':%s,'conf_interval':%s}" %
                (self.l,self.b,self.alpha,self.beta,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, alpha=a, beta=b)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t * b_t + alpha * e_t+1
        self.l1 = self.l * self.b + self.alpha * self.e1

    def trend(self):
        # b_t+1 = b_t + beta * e_t+1 / l_t
        self.b1 = self.b + self.beta * self.e1 / np.maximum(self.l, 0.01)

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        self.b = self.b1
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high]))
        self.pred_interval = (self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high])
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * b_t
        self.y_hat = self.l * self.b
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 11.ETS(A,M,A) - class 5 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AMA(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t * b_t + alpha * e_t+1
        self.l1 = self.l * self.b + self.alpha * self.e1

    def trend(self):
        # b_t+1 = b_t + beta * e_t+1 / l_t
        self.b1 = self.b + self.beta * self.e1 / np.maximum(self.l, 0.001)

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * e_t+1
        self.s1 = self.s[self.m-1] + self.gamma * self.e1

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high]))
        self.pred_interval = (self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high])
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * b_t + s_t-m+1
        self.y_hat = self.l * self.b + self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 12.ETS(A,M,M) - class 5 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_AMM(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=-1
    def error(self, y_observed):
        # e_t+1 = y_observed_t+1 - y_hat_t+1
        self.e1 = y_observed - self.y_hat

    def level(self):
        # l_t+1 = l_t * b_t + alpha * e_t+1 / s_t-m+1
        self.l1 = self.l * self.b + self.alpha * self.e1 / np.maximum(self.s[self.m-1], 0.01)

    def trend(self):
        # b_t+1 = b_t + beta * e_t+1 / (s_t-m+1 * l_t)
        self.b1 = self.b + self.beta * self.e1 / np.maximum(self.s[self.m-1] * self.l, 0.01)

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * e_t+1 / (l_t * b_t)
        self.s1 = self.s[self.m-1] + self.gamma * self.e1 / np.maximum(self.l * self.b, 0.01)

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high]))
        self.pred_interval = (self.y_hat + residuals[quantile_low], self.y_hat + residuals[quantile_high])
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * b_t * s_t-m+1
        self.y_hat = self.l * self.b * self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1

# 13.ETS(A,Md,N)
# 14.ETS(A,Md,A)
# 15.ETS(A,Md,M)

###################################
### Multiplicative Error Models ###
###################################
# 16.ETS(M,N,N) - class 2 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MNN(ETS):
    def __str__(self):
        return ("{'l0':%s,'alpha':%s,'conf_interval':%s}" %
                (self.l,self.alpha,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, alpha=a)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - l_t) / l_t
        self.e1 = (y_observed - self.l) / np.maximum(self.l, 0.0001)

    def level(self):
        # l_t+1 = l_t * (1 + alpha * e_t+1)
        self.l1 = self.l * (1 + self.alpha * self.e1)

    def trend(self):
        pass

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        # update the prediction interval
        var = self.sigma2 * self.y_hat**2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t
        self.y_hat = self.l
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 17.ETS(M,N,A) - class 2 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MNA(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'s0':%s,'alpha':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,ss,self.alpha,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, s0=s0, alpha=a, gamma=g, m=m)

        # object variable
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t + s_t-m+1)) / (l_t + s_t-m+1)
        self.e1 = (y_observed - (self.l + self.s[self.m-1])) / np.maximum((self.l + self.s[self.m-1]), 0.01)

    def level(self):
        # l_t+1 = l_t + alpha * (l_t + s_t-m+1) * e_t+1
        self.l1 = self.l + self.alpha * (self.l + self.s[self.m-1]) * self.e1

    def trend(self):
        pass

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * (l_t + s_t-m+1) * e_t+1
        self.s1 = self.s[self.m-1] + self.gamma * (self.l + self.s[self.m-1]) * self.e1

    def update(self):
        self.l = self.l1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2 * self.y_hat**2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + s_t-m+1
        self.y_hat = self.l + self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 18.ETS(M,N,M) - class 3 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MNM(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'s0':%s,'alpha':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,ss,self.alpha,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, s0=s0, alpha=a, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.s1 = 0
        self.e1 = 0
        self.gamma2 = g**2

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t * s_t-m+1)) / (l_t * s_t-m+1)
        self.e1 = (y_observed - (self.l * self.s[self.m-1])) / np.maximum((self.l * self.s[self.m-1]), 0.01)

    def level(self):
        # l_t+1 = l_t * (1 + alpha * e_t+1)
        self.l1 = self.l * (1 + self.alpha * self.e1)

    def trend(self):
        pass

    def season(self):
        # s_t+1 = s_t-m+1 * (1 + gamma * e_t+1)
        self.s1 = self.s[self.m-1] * (1 + self.gamma * self.e1)

    def update(self):
        self.l = self.l1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        mean_tide2 = (self.y_hat / np.maximum(self.s[self.m-1],0.01))**2
        var = self.s[self.m-1]**2 * (mean_tide2*(1+self.sigma2)*(1+self.gamma2*self.sigma2) - mean_tide2)
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * s_t-m+1
        self.y_hat = self.l * self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 19.ETS(M,A,N) - class 2 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MAN(ETS):
    def __str__(self):
        return ("{'l0':%s,'b0':%s,'alpha':%s,'beta':%s,'conf_interval':%s}" %
                (self.l,self.b,self.alpha,self.beta,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, alpha=a, beta=b)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t + b_t)) / (l_t + b_t)
        self.e1 = (y_observed - (self.l + self.b)) / np.maximum(self.l + self.b, 0.0001)

    def level(self):
        # l_t+1 = (l_t + b_t) * (1 + alpha * e_t+1)
        self.l1 = (self.l + self.b) * (1 + self.alpha * self.e1)

    def trend(self):
        # b_t+1 = b_t + beta * (self.l_t + self.b_t) * e_t+1
        self.b1 = self.b + self.beta * (self.l + self.b) * self.e1

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        self.b = self.b1
        # update the prediction interval
        var = self.sigma2 * self.y_hat**2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + b_t
        self.y_hat = self.l + self.b
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 20.ETS(M,A,A) - class 2 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MAA(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t + b_t + s_t-m+1)) / (l_t + b_t + s_t-m+1)
        self.e1 = (y_observed - (self.l + self.b + self.s[self.m-1])) / np.maximum(self.l + self.b + self.s[self.m-1], 0.0001)

    def level(self):
        # l_t+1 = l_t + b_t + alpha * (l_t + b_t + s_t-m+1) * e_t+1
        self.l1 = self.l + self.b + self.alpha * (self.l + self.b + self.s[self.m-1]) * self.e1

    def trend(self):
        # b_t+1 = b_t + beta * (l_t + b_t + s_t-m+1) * e_t+1
        self.b1 = self.b + self.beta * (self.l + self.b + self.s[self.m-1]) * self.e1

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * (l_t + b_t + s_t-m+1) * e_t+1
        self.s1 = self.s[self.m-1] + self.gamma * (self.l + self.b + self.s[self.m-1]) * self.e1

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2 * self.y_hat**2
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t + b_t + s_t-m+1
        self.y_hat = self.l + self.b + self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 21.ETS(M,A,M) - class 3 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MAM(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variable
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0
        self.gamma2 = g**2

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t + b_t) * s_t-m+1) / ((l_t + b_t) * s_t-m+1)
        self.e1 = (y_observed - (self.l+self.b)*self.s[self.m-1]) / np.maximum((self.l+self.b)*self.s[self.m-1], 0.0001)

    def level(self):
        # l_t+1 = (l_t + b_t) * (1 + alpha * e_t+1)
        self.l1 = (self.l + self.b) * (1 + self.alpha * self.e1)

    def trend(self):
        # b_t+1 = b_t + beta * (l_t + b_t) * e_t+1
        self.b1 = self.b + self.beta * (self.l + self.b) * self.e1

    def season(self):
        # s_t+1 = s_t-m+1 * (1 + gamma * e_t+1)
        self.s1 = self.s[self.m-1] * (1 + self.gamma * self.e1)

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        mean_tide2 = (self.y_hat / np.maximum(self.s[self.m-1],0.01))**2
        var = self.s[self.m-1]**2 * (mean_tide2*(1+self.sigma2)*(1+self.gamma2*self.sigma2) - mean_tide2)
        #self.pred_interval.append(norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var)))
        self.pred_interval = norm.interval(self.conf_interval, loc=self.y_hat, scale=np.sqrt(var))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = (l_t + b_t) * s_t-m+1
        self.y_hat = (self.l + self.b) * self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 22.ETS(M,Ad,N)
# 23.ETS(M,Ad,A)
# 24.ETS(M,Ad,M)


# 25.ETS(M,M,N) - class 4 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MMN(ETS):
    def __str__(self):
        return ("{'l0':%s,'b0':%s,'alpha':%s,'beta':%s,'conf_interval':%s}" %
                (self.l,self.b,self.alpha,self.beta,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, alpha=a, beta=b)

        # object variable
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - l_t * b_t) / (l_t * b_t)
        self.e1 = (y_observed - (self.l * self.b)) / np.maximum(self.l * self.b, 0.0001)

    def level(self):
        # l_t+1 = (l_t * b_t) * (1 + alpha * e_t+1)
        self.l1 = (self.l * self.b) * (1 + self.alpha * self.e1)

    def trend(self):
        # b_t+1 = b_t * (1 + beta * e_t+1)
        self.b1 = self.b * (1 + self.beta * self.e1)

    def season(self):
        pass

    def update(self):
        self.l = self.l1
        self.b = self.b1
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat * (1 + residuals[quantile_low]), self.y_hat * (1 + residuals[quantile_high])))
        self.pred_interval = (self.y_hat * (1 + residuals[quantile_low]), self.y_hat * (1 + residuals[quantile_high]))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * b_t
        self.y_hat = self.l * self.b
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 26.ETS(M,M,A) - class 5 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MMA(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variables
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t * b_t + s_t-m+1)) / (l_t * b_t + s_t-m+1)
        self.e1 = (y_observed - (self.l * self.b + self.s[self.m-1])) / np.maximum(self.l * self.b + self.s[self.m-1], 0.0001)

    def level(self):
        # l_t+1 = (l_t * b_t) + alpha * (l_t * b_t + s_t-m+1) * e_t+1
        self.l1 = (self.l * self.b) + self.alpha * (self.l * self.b + self.s[self.m-1]) * self.e1

    def trend(self):
        # b_t+1 = b_t + beta * (l_t * b_t + s_t-m+1) * e_t+1 / l_t
        self.b1 = self.b + self.beta * (self.l * self.b + self.s[self.m-1]) * self.e1 / np.maximum(self.l, 0.0001)

    def season(self):
        # s_t+1 = s_t-m+1 + gamma * (l_t * b_t + s_t-m+1) * e_t+1
        self.s1 = self.s[self.m-1] + self.gamma * (self.l * self.b + self.s[self.m-1]) * self.e1

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat * (1 + residuals[quantile_low]), self.y_hat * (1 + residuals[quantile_high])))
        self.pred_interval = (self.y_hat * (1 + residuals[quantile_low]), self.y_hat * (1 + residuals[quantile_high]))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * b_t + s_t-m+1
        self.y_hat = self.l * self.b + self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1


# 27.ETS(M,M,M) - class 4 (ref. "Forecasting with exponential smoothing", Rob J. Hyndman)
class ets_MMM(ETS):
    def __str__(self):
        ss = "[%s" % self.s[0]
        if(self.m > 1):
            for i in range(1,self.m):
                ss += ",%s" % self.s[i]
        ss += "]"
        return ("{'l0':%s,'b0':%s,'s0':%s,'alpha':%s,'beta':%s,'gamma':%s,'m':%s,'conf_interval':%s}" %
                (self.l,self.b,ss,self.alpha,self.beta,self.gamma,self.m,self.conf_interval))

    def __init__(self, state0, parameters, conf_interval=0.95):
        l0 = state0[0] # the level value at period t=0
        b0 = state0[1] # the trend value at period t=0
        s0 = state0[2] # the season value at period t=0
        a  = parameters[0] # alpha
        b  = parameters[1] # beta
        g  = parameters[2] # gamma
        m  = parameters[3] # m
        p  = parameters[4] # phi
        ETS.__init__(self, l0=l0, b0=b0, s0=s0, alpha=a, beta=b, gamma=g, m=m)

        # object variable
        self.conf_interval = conf_interval

        # temporary placeholders for the update phase
        self.l1 = 0
        self.b1 = 0
        self.s1 = 0
        self.e1 = 0

    ###############
    # Update phase
    # before update: the state is at period t=0
    def error(self, y_observed):
        # e_t+1 = (y_observed_t+1 - (l_t * b_t * s_t-m+1)) / (l_t * b_t * s_t-m+1)
        self.e1 = (y_observed - (self.l * self.b * self.s[self.m-1])) / np.maximum(self.l * self.b * self.s[self.m-1], 0.0001)

    def level(self):
        # l_t+1 = (l_t * b_t) * (1 + alpha * e_t+1)
        self.l1 = (self.l * self.b) * (1 + self.alpha * self.e1)

    def trend(self):
        # b_t+1 = b_t * (1 + beta * e_t+1)
        self.b1 = self.b * (1 + self.beta * self.e1)

    def season(self):
        # s_t+1 = s_t-m+1 * (1 + gamma * e_t+1)
        self.s1 = self.s[self.m-1] * (1 + self.gamma * self.e1)

    def update(self):
        self.l = self.l1
        self.b = self.b1
        self.s[0] = self.s1
        np.roll(self.s,1) # right-shift the array s
        # update the prediction interval
        var = self.sigma2
        residuals = np.random.normal(loc=0, scale=np.sqrt(var), size=self.sample_size)
        residuals.sort()
        quantile_low  = int(self.sample_size * (1-self.conf_interval)/2)
        quantile_high = self.sample_size - quantile_low - 1
        #self.pred_interval.append((self.y_hat * (1 + residuals[quantile_low]), self.y_hat * (1 + residuals[quantile_high])))
        self.pred_interval = (self.y_hat * (1 + residuals[quantile_low]), self.y_hat * (1 + residuals[quantile_high]))
    # after update: the state is at period t=1,
    #               shift the time window => the state is at t=0 and ready for the prediction phase

    ####################
    # Prediction phase
    # before prediction: the state is at period t=0
    def predict(self):
        # predict y_hat for the next period t=1
        # y_hat_t+1 = l_t * b_t * s_t-m+1
        self.y_hat = self.l * self.b * self.s[self.m-1]
        return self.y_hat
    # after prediction: the state is at period t=0, y_hat at t=1

# 28.ETS(M,Md,N)
# 29.ETS(M,Md,A)
# 30.ETS(M,Md,M)

