import numpy as np
import json
import os.path
import copy
import matplotlib.pyplot as plt

from time_series import time_series
from benchmark_model import benchmark
from ets_models import ets_AAN, ets_AMN, ets_MNN, ets_MAN, ets_MMN, ets_ANA, ets_AAA, ets_AMA, ets_AMM, ets_ANN, ets_ANM, ets_AAM, ets_MNA, ets_MNM, ets_MAA, ets_MAM, ets_MMA, ets_MMM

from local_modeling import local_modeling

#######################################################################################
# Configuration for automation: 
# . seperate dataset to 2 portions: training-testing, validation
# . estimate suitable window size: training-testing phase
#    - SESE tool
# . estimate initial state: heuristics, fixed
# . estimate optimal parameters: validation phase
#    - SSE tool
# . do forecasting: validation phase
#######################################################################################


#######################################################################################
# master parameters for the whole program
#
fname = './data/AAPL_2015_2016_2017.csv'
price = 'adj_close' # do forecasting on the adjusted close price
confd = {'1sigma':0.6827, '2sigma':0.9545,'3sigma':0.9973} # confidence intervals
CI    = '3sigma' # forecast with 3-sigma confidence

# Load the whole time series
ts = time_series(fname)

#######################################################################################
# master parameters for the local modeling framework
#day_len = 504 # first 2 years
#day_len = 252 # first one year
#day_len = 191 # first nine months
#day_len = 122 # first six months
#day_len = 61 # first three months
#day_len = 40 # first two months
#day_len = 20 # first month

# 5 datasets to estimate the suitable window size in the training-testing phase
#day_start   = [0,63,126,189,252]
day_start   = [0,63]
day_len     = 63

# dataset to do forecasting in the validation phase
validate_day_start = 504
validate_day_len   = 191

# hope the suitable window size is one of the following lengths
segment_len = [2,10,18,20,25]


# Models declaration, all models which may be used in the program are listed here
model_bmk = {'Benchmark':('BMK',benchmark)}

# all ETS models
'''
model_ets = {'ANN':{'type':'ETS','class':ets_ANN,'seglen':segment_len},
             'ANA':{'type':'ETS','class':ets_ANA,'seglen':segment_len},
             'ANM':{'type':'ETS','class':ets_ANM,'seglen':segment_len},
             'AAN':{'type':'ETS','class':ets_AAN,'seglen':segment_len},
             'AAA':{'type':'ETS','class':ets_AAA,'seglen':segment_len},
             'AAM':{'type':'ETS','class':ets_AAM,'seglen':segment_len},
             'AMN':{'type':'ETS','class':ets_AMN,'seglen':segment_len},
             'AMA':{'type':'ETS','class':ets_AMA,'seglen':segment_len},
             'AMM':{'type':'ETS','class':ets_AMM,'seglen':segment_len},
             'MNN':{'type':'ETS','class':ets_MNN,'seglen':segment_len},
             'MNA':{'type':'ETS','class':ets_MNA,'seglen':segment_len},
             'MNM':{'type':'ETS','class':ets_MNM,'seglen':segment_len},
             'MAN':{'type':'ETS','class':ets_MAN,'seglen':segment_len},
             'MAA':{'type':'ETS','class':ets_MAA,'seglen':segment_len},
             'MAM':{'type':'ETS','class':ets_MAM,'seglen':segment_len},
             'MMN':{'type':'ETS','class':ets_MMN,'seglen':segment_len},
             'MMA':{'type':'ETS','class':ets_MMA,'seglen':segment_len},
             'MMM':{'type':'ETS','class':ets_MMM,'seglen':segment_len}}
'''
model_ets = {'ANN':{'type':'ETS','class':ets_ANN,'seglen':segment_len},
             'MNN':{'type':'ETS','class':ets_MNN,'seglen':segment_len}}
#model_ets = {'ANN':{'type':'ETS','class':ets_ANN,'seglen':segment_len}}
#model_ets = {'MNN':{'type':'ETS','class':ets_MNN,'seglen':segment_len}}

model_all = dict(model_bmk)
model_all.update(model_ets)


#######################################################################################
################################### Training phase ####################################
# Purpose of this phase: choose the most suitable window size (segment length) for each model
print("==================")
print("= Training phase =")
print("==================")

# Placeholder to store outputs of the training phase, ie. the sese performances
train_sese   = {} # the sese values, for ploting only.
cutting_slen = {} # the segment length at the cutting points of all models,
n_curves = 2      # cutting point where at least 2 out of 5 SESE curves have tendency to increase

for name,info in model_ets.iteritems():
    mdl = dict(model_bmk)
    mdl.update({name:(info['type'],info['class'])})
    slen_list = info['seglen']
    train_sese[name] = {} 
    # all points on the sese curves have zero probability to be the cutting point.
    cutting_prob       = [0 for i in range(len(slen_list))]
    cutting_slen[name] = {}      # each model may have different cutting point.
    stop               = False   # implement the early stoping in training process

    print("= Train on multiple datasets =")
    dataset_idx = 0
    while (dataset_idx < len(day_start) and stop == False):
        train_sese[name][dataset_idx] = np.array([None for i in range(len(segment_len))])
        prev_sese = float("inf")

        print("--Train on multiple window sizes--")
        slen_idx = 0
        while (slen_idx < len(slen_list) and stop == False):
            # Run model with the training dataset and predefined segment length
            train_modeling = local_modeling('train', mdl, price, ts, day_start[dataset_idx],\
                                            day_len, slen_list[slen_idx])
            train_modeling.optimize_predict(confd[CI])

            # Dump everything to files
            train_modeling.write_to_file()
            train_modeling.plot_to_file(CI)

            # Get SESE performance to evaluate the training process
            sese = train_modeling.get_sese()[name]
            train_sese[name][dataset_idx][slen_idx] = sese

            # early stoping the training process
            cutting_slen[name]['nonOptimal'] = slen_list[slen_idx]
            cutting_slen[name]['Optimal']    = slen_list[slen_idx]
            if sese >= prev_sese:
                # the previous performance may be the cutting point
                cutting_prob[slen_idx-1] += 1
                if cutting_prob[slen_idx-1] >= n_curves:
                    stop = True # stop the whole training process for this model
                    cutting_slen[name]['Optimal'] = slen_list[slen_idx-1]
            prev_sese = sese

            slen_idx += 1
        dataset_idx += 1
    


# create folder to contain all outputs in the training phase
if not os.path.exists('output/'+ 'train'):
    os.makedirs('output/' + 'train')

# write to file the cutting points
outfname = './output/train/' + 'train' + '_segmentlen'
with open(outfname + '_cutting.txt', 'w') as file:
    file.write(json.dumps(cutting_slen))

# write to file the SESE performance
outfname = './output/train/' + 'train' + '_segmentlen'
with open(outfname + '_sese.txt', 'w') as file:
    for name in list(model_ets.keys()):
        for dataset_idx in list(train_sese[name].keys()):
            file.write('model {0}, dataset {1}, sese = {2} \n'.format(name, day_start[dataset_idx], train_sese[name][dataset_idx]))

# plot
for name in list(model_ets.keys()):
    print("cutting point of model {} is {}".format(name, cutting_slen[name]['Optimal']))
    outfname = './output/train/' + 'train' + '_performance_sese' + '_ets' + name
    fig = plt.figure()
    plt.xlabel('Segment length')
    plt.ylabel('SESE measurement')
    plt.title('SESE curve of ' + name)
    for dataset_idx in list(train_sese[name].keys()):
        print('model {}, dataset {}, sese = {}'.format(name, day_start[dataset_idx], train_sese[name][dataset_idx]))
        plt.plot(segment_len, train_sese[name][dataset_idx], marker='o', label=str(day_start[dataset_idx]))
        plt.xticks(segment_len)
        plt.legend()
        plt.axvspan(cutting_slen[name]['Optimal']-1, cutting_slen[name]['Optimal']+1, color='grey', alpha=0.5)
    fig.savefig(outfname + '.png') 
    plt.close(fig)


#######################################################################################
################################## Validation phase ###################################
print("====================")
print("= Validation phase =")
print("====================")
# Placeholders to get back performances of each model
optimal_sese    = {}
optimal_sse     = {}
nonOptimal_sese = {}
nonOptimal_sse  = {}

# get the models to be validated together with their suitable window sizes
for name,info in model_ets.iteritems():
    mdl = dict(model_bmk)
    mdl.update({name:(info['type'],info['class'])})

    # compare the model performances in two cases to evaluate the hypothesis
    # about the optimal segment length.
    optimal_seglen    = cutting_slen[name]['Optimal']
    nonOptimal_seglen = cutting_slen[name]['nonOptimal']

    ################################################
    # Run the model with optimal segment length
    print("validate_optimal {}".format(name))
    validate_modeling = local_modeling('validate_optimal', mdl, price, ts, validate_day_start,\
                                        validate_day_len, optimal_seglen)
    validate_modeling.optimize_predict(confd[CI])

    # dump everything to files
    validate_modeling.write_to_file()
    validate_modeling.plot_to_file(CI)

    # Get sese & sse performance
    optimal_sese[name] = validate_modeling.get_sese()[name]
    optimal_sse[name]  = validate_modeling.get_sse()[name]


    ################################################
    # Run the model with non-optimal segment length
    if(nonOptimal_seglen != optimal_seglen):
        print("validate_nonoptimal {}".format(name))
        validate_modeling = local_modeling('validate_nonoptimal', mdl, price, ts, validate_day_start,\
                                            validate_day_len, nonOptimal_seglen)
        validate_modeling.optimize_predict(confd[CI])

        # dump everything to files
        validate_modeling.write_to_file()
        validate_modeling.plot_to_file(CI)

        # Get sese & sse performance
        nonOptimal_sese[name] = validate_modeling.get_sese()[name]
        nonOptimal_sse[name]  = validate_modeling.get_sse()[name]



# create folder to contain all outputs in the validation phase
if not os.path.exists('output/'+ 'validate_optimal'):
    os.makedirs('output/' + 'validate_optimal')
if not os.path.exists('output/'+ 'validate_nonoptimal'):
    os.makedirs('output/' + 'validate_nonoptimal')

outfname = './output/validate_optimal/' + 'validate' + '_optimal_sese'
with open(outfname + '.txt', 'w') as file:
    file.write(json.dumps(optimal_sese))

outfname = './output/validate_optimal/' + 'validate' + '_optimal_sse'
with open(outfname + '.txt', 'w') as file:
    file.write(json.dumps(optimal_sse))

outfname = './output/validate_nonoptimal/' + 'validate' + '_nonOptimal_sese'
with open(outfname + '.txt', 'w') as file:
    file.write(json.dumps(nonOptimal_sese))

outfname = './output/validate_nonoptimal/' + 'validate' + '_nonOptimal_sse'
with open(outfname + '.txt', 'w') as file:
    file.write(json.dumps(nonOptimal_sse))



#######################################################################################
# to evaluate the consistency of the model wrt the metrics see and sese.
# The consistent model has these two measurements be consistent, not sponstaneous
# over various datasets.
print("===================")
print("= Evaluate models =")
print("===================")
# TODO


