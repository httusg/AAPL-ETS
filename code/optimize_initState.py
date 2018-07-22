# heuristics: initial values alpha=0.1, beta=0.01, gamma=0.01, phi=0.99

def optimize_initState(y, name='ANN', method='heuristic'):
    '''
    y     : the dataset w/o time index used for optimization
    name  : the string name of the model applying on the y dataset
    method: 'heuristic', 'bkcasting', 'decomp'
    '''

    if (method == 'bkcasting'): # backcasting
        #TODO
        pass
    elif (method == 'decomp'): # decomposition
        #TODO
        pass
    else: # heuristic
        l0 = y[0]

        # ETS models which become the benchmark model under the common initial parameters
        # are belong to the same group.
        model_ets_group1 = ['ANN','ANM','AAdN','AMA','MMA','AMN','AMM','MNN','MNM','MMN','MMM']
        model_ets_group2 = ['ANA','AAN','AAA','MNA','MAN','MAA']
        model_ets_group3 = ['AAM','MAM']

        if (name in model_ets_group1):
            b0 = 1
            s0 = 1
        elif(name in model_ets_group2):
            b0 = 1.0 
            s0 = 0.0 
        elif(name in model_ets_group3):
            b0 = 0.0 
            s0 = 1
        else: # default
            b0 = 0.3
            s0 = 0.3

    return (l0, b0, s0)

