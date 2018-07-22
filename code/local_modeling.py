import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import json
import os.path

from helper import optimize, predict, add_output_columns

# implement local modeling method
class local_modeling():
    def __init__(self, phase_name, model_list, ts_name, time_series, day_start, day_len, segment_len):
        self.phase_name  = phase_name
        self.model_list  = model_list
        self.ts          = time_series
        self.day_start   = day_start
        self.day_len     = day_len
        self.segment_len = segment_len
        self.ts_name     = ts_name  # name of the time series
        print("models:{} day_start:{} day_len:{} segment_len:{}".format(list(model_list.keys()),\
                                                                        day_start, day_len, segment_len))

        ### Prepare dataset input ###
        # Reset the window size and position
        self.ts.reset_window(self.day_start, self.day_len, self.segment_len)
        # Get the first segment to do prediction
        self.series = self.ts.get_next_segment()
        # Get dataset for forecasting w/o time index
        self.y = np.squeeze(self.series.values)

        # Create dataframe df to store prediction results from the selected models
        self.df = self.series.copy(deep=True)
        self.df = add_output_columns(self.df, self.model_list)

        # folder to contain all outputs
        if not os.path.exists('output/'+ phase_name):
            os.makedirs('output/' + phase_name)

    def optimize_predict(self, confidence_interval):
        while (not self.series.empty):
            # 1. Optimize initial state by heuristics
            #    Optimize parameters by exhaustive searching technique
            #    m=0 to optimize m parameter as well.
            #    Optimal m is usually equal to 1 for most of the models, 
            #    could put m=1 to by pass optimizing m (reduce run time much)
            state0, opPara = optimize(self.y, self.model_list, m=0)

            # 2. Do prediction
            prediction = predict(self.y, self.model_list, state0=state0, parameters=opPara, conf_interval=confidence_interval)

            # a. Store prediction results and the optimal parameters
            outcome = {} # all outcomes in one prediction period, ie. one row in the df

            for name in list(prediction.keys()):
                outcome[name]          = prediction[name][0] # y_hat
                outcome[name+'_lower'] = prediction[name][1] # interval
                outcome[name+'_upper'] = prediction[name][2] # interval

            for name in list(self.model_list.keys()):
                if (name is not "Benchmark"):
                    outcome[name+'_alpha'] = opPara[name][0]
                    outcome[name+'_beta']  = opPara[name][1]
                    outcome[name+'_gamma'] = opPara[name][2]
                    outcome[name+'_m']     = opPara[name][3]

            # b. Get the next segment for the next prediction period
            self.series = self.ts.get_next_segment()

            # 3. Measure performances sqe & sse until current period
            if (not self.series.empty):
               self.y = np.squeeze(self.series.values)
               y_observed = self.y[len(self.y)-1]
               outcome[self.ts_name] = y_observed

               for name in list(prediction.keys()):
                   y_hat = prediction[name][0]
                   error = y_observed - y_hat
                   sse   = self.df[name+'_sse'][len(self.df)-1] # latest sse
                   outcome[name+'_sqe'] = error**2
                   outcome[name+'_sse'] = sse + error**2
                   outcome[name+'_acf'] = error # to check autocorrelation

            # c. Append the outcome to df
            if (not self.series.empty): # time index for tomorrow is available
                tomorrow = self.series.index[len(self.series)-1] 
            else: # no time index for tomorrow, assuming tomorrow is the next day
                today = self.df.index[len(self.df)-1]
                tomorrow = today + datetime.timedelta(days=1)

            row = pd.DataFrame([outcome])    
            row.reset_index(inplace=False)
            row['date'] = tomorrow
            row = row.set_index('date')
            self.df = self.df.append(row, sort=True)

        # End of prediction, calculate SESE
        self.sese = {} # store SESE values of all models
        for name in list(self.model_list.keys()):
            self.df[name+'_ese'] = self.df[name+'_sqe'] - self.df['Benchmark'+'_sqe']
            self.sese[name] = self.df[name+'_ese'].sum()

        #print("=self.df=")
        #print(self.df)

    def get_sse(self):
        sse = {}
        for name in list(self.model_list.keys()):
            sse[name] = self.df[name+'_sse'][len(self.df)-2] # there is no sse at the latest prediction
        return sse

    def get_sese(self):
        return self.sese


    def write_to_file(self):
        #######################################################################################
        # 3.Write to file
        # time series input, predictions, prediction intervals,
        # performance measurements, parameter estimates of all models
        mdlNames = ""
        for name in list(self.model_list.keys()):
            if (name is not "Benchmark"):
                mdlNames += '_' + name

        outfname = './output/' + self.phase_name + '/' + \
                   self.phase_name + '_dataset' + str(self.day_start) + \
                   '_segment' + str(self.segment_len) + \
                   '-' + str(self.day_len) + 'days' + '_ets' + mdlNames
        self.df.to_csv(outfname + '.csv')

        # SESE measurements of all models
        with open(outfname + '_sese.txt', 'w') as file:
            file.write(json.dumps(self.sese))


    def plot_to_file(self, CI):
        #######################################################################################
        # 4. Plotting
        for name in list(self.model_list.keys()):
            if(name is not "Benchmark"):
                outfname = './output/' + self.phase_name + '/' + \
                           self.phase_name + '_dataset' + str(self.day_start) + \
                           '_segment' + str(self.segment_len) + \
                           '-' + str(self.day_len) + 'days' + '_ets' + name

                # 4.1. Forecasting
                shown_curves = [self.ts_name, 'Benchmark', name]
                fig_pred = plt.figure(1)
                ax_pred  = plt.axes()
                self.df[shown_curves].plot(grid=True, figsize=(8,5), title="Forecast AAPL", label='Forecast', ax=ax_pred)
                fig_pred.savefig(outfname + '_forecast.png') 
                plt.close(fig_pred)

                # 4.2. Confident intervals
                shown_curves = [name + '_lower', name + '_upper']
                fig_intv = plt.figure(2)
                ax_intv  = plt.axes()
                self.df[[self.ts_name]].plot(grid=True, figsize=(8,5), title="Confident Interval", label='CI', ax=ax_intv)
                self.df[shown_curves].plot(grid=True, linestyle='dashed', ax=ax_intv)
                fig_intv.savefig(outfname + '_CI' + CI + '.png') 
                plt.close(fig_intv)

                # 4.3. Autocorrelation
                shown_curves = ['Benchmark_acf', name + '_acf']
                fig_acf = plt.figure(3)
                ax_acf  = plt.axes()
                self.df[shown_curves].plot(grid=True, figsize=(8,5), title="Autocorrelation", label='ACF', ax=ax_acf)
                fig_acf.savefig(outfname + '_acf.png') 
                plt.close(fig_acf)

                # 4.4. Performance
                shown_curves = ['Benchmark_sse', name + '_sse']
                fig_sse = plt.figure(4)
                ax_sse  = plt.axes()
                #ax_sse.axhline(y=0, color='lightgrey', linestyle='-') # baseline to compare prime values
                self.df[shown_curves].plot(grid=True, figsize=(8,5), title="Sum of Squared Errors", label='SSE', ax=ax_sse)
                #self.df[bmk_p1].plot(style='k*' , ax=ax_sse)
                fig_sse.savefig(outfname + '_sse.png') 
                plt.close(fig_sse)

                # 4.5. Parameter estimates
                shown_curves = [name+'_alpha', name+'_beta', name+'_gamma', name+'_m']
                fig_para = plt.figure(5)
                ax_para  = plt.axes()
                self.df[shown_curves].plot(grid=True, figsize=(8,5), title="Parameter Estimates", label='Para', ax=ax_para)
                fig_para.savefig(outfname + '_para.png') 
                plt.close(fig_para)

                #ax.legend(loc='upper left') 
                #plt.show()

