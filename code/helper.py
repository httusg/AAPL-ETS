import numpy as np
import json
from optimize_initState import optimize_initState
from optimize_parameters import optimize_model


def add_output_columns(df, model_list):
    for name,_ in model_list.iteritems():
        # prediction results
        df[name]          = [np.nan for i in range(len(df))]
        df[name+'_lower'] = [np.nan for i in range(len(df))]
        df[name+'_upper'] = [np.nan for i in range(len(df))]

        # initial parameters
        if (name is not "Benchmark"):
            df[name+'_alpha'] = [np.nan for i in range(len(df))]
            df[name+'_beta']  = [np.nan for i in range(len(df))]
            df[name+'_gamma'] = [np.nan for i in range(len(df))]
            df[name+'_m']     = [np.nan for i in range(len(df))]

        # performance
        df[name+'_acf'] = [np.nan for i in range(len(df))] # to check autocorrelation
        df[name+'_sse'] = [0      for i in range(len(df))]
        df[name+'_sqe'] = [0      for i in range(len(df))] # squared errors (sqe) = first derivative of sse series
        df[name+'_ese'] = [0      for i in range(len(df))] # error of squared errors (ese) = model's sqe - BMK's sqe

    return df


def get_plotted_columns_name(model_list):
    modelNames = list(model_list.keys())

    # list of column names containing prediction intervals (all except benchmark model)
    intervals  = [name+'_lower' for name in modelNames if (name is not "Benchmark")] 
    intervals += [name+'_upper' for name in modelNames if (name is not "Benchmark")]

    # list of column names containing parameters (alpha, beta, gamma, m)
    parameters  = [name+'_alpha' for name in modelNames if (name is not "Benchmark")]
    parameters += [name+'_beta'  for name in modelNames if (name is not "Benchmark")]
    parameters += [name+'_gamma' for name in modelNames if (name is not "Benchmark")]
    parameters += [name+'_m'     for name in modelNames if (name is not "Benchmark")]

    # list of column names containing performance curves of all models.
    acf    = [name+'_acf' for name in modelNames]  # to check autocorrelation
    sse    = [name+'_sse' for name in modelNames]
    bmk_p1 = [name+'_ese' for name in modelNames]  # diff of model's sqe from BMK's sqe

    return intervals, parameters, acf, sse, bmk_p1


def write_ETS_model_to_file(phase_name, model_list):
    # Eliminate models with no window size
    mdl_list = {}
    for name,(_,_,s,e) in model_list.iteritems():
        if (s is not ''):
            mdl_list[name] = (s,e)

    # Write to file the list of models together with its suitable window size and sese
    mdlNames = ""
    for name,_ in mdl_list.iteritems():
        mdlNames += '_' + name
    outfname = './output/' + phase_name + '_ets' + mdlNames

    with open(outfname + '_window.txt', 'w') as file:
        file.write(json.dumps(mdl_list))


def do_optimize(y, model, name, m=1):
    '''
    y    : the dataset w/o time index
    model: the class of the model
    name : the string name of the model
    m: the number of seasons,
       if m is 0, optimizing m is required
    '''
    # 2.1.Optimize initial state
    state0 = optimize_initState(y, name=name)
    #print("model {}, optimal state l0,b0,s0 {}".format(name, state0))

    # 2.2.Optimize the model/parameters
    parameters = optimize_model(y, model, name, state0, m=m)
    #print('model {}, optimal parameters alpha,beta,gamma,m {}'.format(name, parameters))

    return state0, parameters


def optimize(y, model_list, m=1):
    '''
    y         : the dataset w/o time index
    model_list: the list of models to be optimized

    m: the number of seasons,
       if m is 0, optimizing m is required,
    '''
    state0 = {} # initial state of all models
    opPara = {} # optimal parameters

    for name,(_,model) in model_list.iteritems():
        # optimize both state and parameters for each model
        state, para = do_optimize(y, model, name, m=m)
        state0[name] = state
        opPara[name] = para

    return state0, opPara


def predict(y, model_list, state0, parameters, conf_interval=0.683):
    '''
    y         : dataset w/o time index for forecasting
    model_list: the list of models to do forecasting
    state0    : the list of initial states of the models
    parameters: the list of initial parameters of the models
    conf_interval : confident interval to get prediction intervals
    '''

    prediction = {} # The prediction of the first period of the time series should be NaN.

    for name,(_,model) in model_list.iteritems():
        # Create model with initial condition.
        mdl = model(state0[name], parameters[name], conf_interval=conf_interval)

        # Autoregression process to reach the latest period t
        for i in range(1,len(y)): 
            mdl.predict_observe(y[i])

        # Do forecasting for the future period t+1
        y_hat = mdl.predict()
        (y_lower, y_upper) = mdl.pred_interval
        prediction[name] = (y_hat, y_lower, y_upper)

    return prediction


