from ets import ETS

#########################################################################
### Benchmark model: naive prediction (right-shift),
### wrap the benchmark model by the ETS class
### to reuse the code calculating performance
#########################################################################
class benchmark(ETS):
    def __str__(self):
        return ("{'l0': %s}" % self.l)

    def __init__(self, state0, parameters=0, conf_interval=0, train_validation=0):
        '''
        l0 : the observed value at the period t=0
        '''
        l0 = state0[0]
        ETS.__init__(self, l0=l0)

        # temporary placeholder for the update phase
        self.y_observed = l0

    ###############
    # Update phase, do nothing but save the observation
    def error(self, y_observed):
        self.y_observed = y_observed

    def level(self):
        pass
    def trend(self):
        pass
    def season(self):
        pass
    def update(self):
        self.var = 0
        #self.pred_interval.append((self.y_hat,self.y_hat))
        self.pred_interval = (self.y_hat, self.y_hat)

    ####################
    # Prediction phase, the future value = the observed value
    def predict(self):
        self.y_hat = self.y_observed
        return self.y_hat


