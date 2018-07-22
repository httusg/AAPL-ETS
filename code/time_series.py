import pandas as pd

class time_series():
    def __init__(self, fname):
        '''
        fname: file name containing the whole time series
        '''
        self.df = pd.read_csv(fname)
        self.df = self.df[self.df['ticker']=="AAPL"]
        #print("=whole series=")
        #print(df)

    def reset_window(self, tick_start=0, tick_len=1, slen=1):
        '''
        tick_start : the first tick in the series
        tick_len   : the number of ticks (eg. days) used for the experiment
                     20 for one month,
                     61 for three months,
                     252 for one year,
                     504 for two years,
                     753 for three years
        slen       : the number of ticks in each segment
        '''
        self.seglen = slen           # segment length, the number of ticks in one segment.
        self.segoff = tick_start     # the begining of the segment in the whole time series
        self.limit  = tick_start + tick_len # the limit in series at which the segment should not across
        self.effective_len = self.limit - self.seglen
        
    def get_next_segment(self):
        if(self.segoff + self.seglen <= self.limit):
            remaining_len = self.limit - self.segoff - self.seglen
            if (self.effective_len != 0):
                percent = 1 - float(remaining_len) / self.effective_len
            else:
                percent = 1
            #print("")
            #print("progress ... {}%, {} remaining".format(percent*100, remaining_len))

            series = self.df[(self.df['date'] >= self.df['date'][self.segoff]) &
                             (self.df['date'] <= self.df['date'][self.segoff + self.seglen - 1])]

            # Extract time series input
            series = series.filter(items=['adj_close'])

            # Convert index to DateTimeIndex, required by the lib function seasonal_decompose()
            series.reset_index(inplace=False)
            series['date'] = pd.to_datetime(self.df['date'])
            series = series.set_index('date')

            self.segoff += 1
        else:
            series = pd.DataFrame()

        #print(series)
        return series
