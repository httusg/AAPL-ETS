import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALPHA = 1 # the class ETS hits overflow in accumulated 'error' variable when 1<alpha<2
PHI   = 1

##############################################################################
# optimizer for alpha parameter
def optimizer_ets_a(y, ets_model, state0):
    '''
    y         : the dataset w/o time index
    ets_model : ETS model to be optimized.
    state0    : the initial state to begin optimization
    '''
    sse = float("inf")
    #fig = plt.figure()
    #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

    for a in np.arange(0.0, ALPHA+0.1, 0.1): # 0 <= alpha <= 1
        para = (a,0,0,0,0) # alpha, beta, gamma, m, phi
        ets = ets_model(state0=state0, parameters=para)
        #for i in range(0,len(y)):
        for i in range(1,len(y)):
            ets.predict_observe(y[i])
        # store performance coressponding to the alpha
        if (ets.sse < sse):
            sse = ets.sse
            alpha = a
        # plot
        #ax.scatter(x=a, y=ets.sse, c='b', marker='o')

    #ax.set_xlabel('Alpha')
    #ax.set_ylabel('SSE')
    #print("alpha={}".format(alpha))
          # alpha, beta, gamma, m, phi
    return (alpha,    0,     0, 0,   0) 


##############################################################################
# optimizer for alpha & beta parameters
def optimizer_ets_ab(y, ets_model, state0):
    '''
    y         : the dataset w/o time index
    ets_model : ETS model to be optimized.
    state0    : the initial state to begin optimization
    '''
    sse = float("inf")
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

    for a in np.arange(0.0, ALPHA+0.1, 0.1): # 0 <= alpha <= 1
        for b in np.arange(0.0, a+0.01, 0.01): # 0 <= beta <= alpha
            para = (a,b,0,0,0) # alpha, beta, gamma, m, phi
            ets = ets_model(state0=state0, parameters=para)
            #for i in range(0,len(y)):
            for i in range(1,len(y)):
                ets.predict_observe(y[i])
            # store performance coressponding to the alpha n beta
            if (ets.sse < sse):
                sse = ets.sse
                alpha = a
                beta  = b
            # plot
            #ax.scatter(xs=a, ys=b, zs=ets.sse, c='b', marker='o')

    #ax.set_xlabel('Alpha')
    #ax.set_ylabel('Beta')
    #ax.set_zlabel('SSE')
    #print("alpha={} beta={}".format(alpha,beta))
          # alpha, beta, gamma, m, phi
    return (alpha, beta,     0, 0,   0)


##############################################################################
# optimizer for alpha, gamma & m parameters
def optimizer_ets_ag(y, ets_model, state0, m):
    '''
    y         : the dataset w/o time index
    ets_model : ETS model to be optimized.
    state0    : the initial state to begin optimization
    m         : the period of the seasonality, m=4 for quarterly, 12 for monthly.
                if m=0, optimizing m is required.
    '''
    sse = float("inf")
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

    for a in np.arange(0.0, ALPHA+0.1, 0.1): # 0 <= alpha <= 1
        for g in np.arange(0.0, (ALPHA-a)+0.01, 0.01): # 0 <= gamma <= ALPHA-alpha
            if (m==0): # optimizing m is required
                for mm in range(1,13,1): #  1<= mm <= 12
                    para = (a,0,g,mm,0) # alpha, beta, gamma, m, phi
                    ets = ets_model(state0=state0, parameters=para)
                    #for i in range(0,len(y)):
                    for i in range(1,len(y)):
                        ets.predict_observe(y[i])

                    # store performance coressponding to the alpha n beta
                    if (ets.sse < sse):
                        sse = ets.sse
                        alpha = a
                        gamma = g
                        season = mm
                        #print("best ets_ag a={} g={} m={} sse={}".format(a, g, mm, sse))
                    #elif (g%0.4 == 0.0):
                    #    print("ets_ag a={} g={} m={} sse={}".format(a, g, mm, ets.sse))

                    # plot
                    #ax.scatter(xs=a, ys=g, zs=ets.sse, c='b', marker='o')

            else: # use m input to optimize there other parameters.
                para = (a,0,g,m,0) # alpha, beta, gamma, m, phi
                ets = ets_model(state0=state0, parameters=para)
                #for i in range(0,len(y)):
                for i in range(1,len(y)):
                    ets.predict_observe(y[i])

                # store performance coressponding to the alpha n beta
                if (ets.sse < sse):
                    sse = ets.sse
                    alpha = a
                    gamma = g
                    season = m
                    #print("fixed best ets_ag a={} g={} m={} sse={}".format(a, g, m, sse))
                #elif (g%0.4 == 0.0):
                #    print("fixed ets_ag a={} g={} m={} sse={}".format(a, g, m, ets.sse))

                # plot
                #ax.scatter(xs=a, ys=g, zs=ets.sse, c='b', marker='o')

    #ax.set_xlabel('Alpha')
    #ax.set_ylabel('Gamma')
    #ax.set_zlabel('SSE')
    #print("alpha={} gamma={} m={}".format(alpha, gamma, season))
          # alpha, beta, gamma,      m, phi
    return (alpha,    0, gamma, season,   0) 


##############################################################################
# optimizer for alpha, beta & phi parameters
def optimizer_ets_abp(y, ets_model, state0):
    '''
    y         : the dataset w/o time index
    ets_model : ETS model to be optimized.
    state0    : the initial state to begin optimization
    '''
    sse = float("inf")
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

    for a in np.arange(0.0, ALPHA+0.1, 0.1): # 0 <= alpha <= 1
        for b in np.arange(0.0, a+0.01, 0.01): # 0 <= beta <= alpha
            for p in np.arange(0.0, PHI+0.01, 0.01): # 0 <= phi <= 1
                para = (a,b,0,0,p) # alpha, beta, gamma, m, phi
                ets = ets_model(state0=state0, parameters=para)
                #for i in range(0,len(y)):
                for i in range(1,len(y)):
                    ets.predict_observe(y[i])
                # store performance coressponding to the alpha n beta
                if (ets.sse < sse):
                    sse = ets.sse
                    alpha = a
                    beta  = b
                    phi   = p
                # plot
                #ax.scatter(xs=a, ys=b, zs=ets.sse, c='b', marker='o')

    #ax.set_xlabel('Alpha')
    #ax.set_ylabel('Beta')
    #ax.set_zlabel('SSE')
    #print("alpha={} beta={} phi={}".format(alpha,beta,phi))
          # alpha, beta, gamma, m, phi
    return (alpha, beta,     0, 0, phi)


##############################################################################
# optimizer for alpha, beta, gamma & m parameters
def optimizer_ets_abg(y, ets_model, state0, m):
    '''
    y         : the dataset w/o time index
    ets_model : ETS model to be optimized.
    state0    : the initial state to begin optimization
    m         : the period of the seasonality, m=4 for quarterly, 12 for monthly,
                if m=0, optimizing m is required.
    '''
    sse = float("inf")
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

    for a in np.arange(0.0, ALPHA+0.1, 0.1): # 0 <= alpha <= 1
        for b in np.arange(0.0, a+0.01, 0.01): # 0 <= beta <= alpha
            for g in np.arange(0.0, (ALPHA-a)+0.01, 0.01): # 0 <= gamma <= ALPHA-alpha
                if (m==0): # optimizing m is required
                    for mm in range(1,13,1): #  1<= mm <= 12
                        para = (a,b,g,mm,0) # alpha, beta, gamma, m, phi
                        ets = ets_model(state0=state0, parameters=para)
                        #for i in range(0,len(y)):
                        for i in range(1,len(y)):
                            ets.predict_observe(y[i])

                        # store performance coressponding to the alpha n beta
                        if (ets.sse < sse):
                            sse = ets.sse
                            alpha = a
                            beta  = b
                            gamma = g
                            season = mm
                            #print("best ets_abg a={} b={} g={} m={} sse={}".format(a, b, g, mm, sse))
                        #elif(a%0.2 == 0) and (b%0.1 == 0) and (g%0.3 == 0):
                        #    print("ets_abg a={} b={} g={} m={} sse={}".format(a, b, g, mm, ets.sse))

                        # plot
                        #ax.scatter(xs=a, ys=g, zs=ets.sse, c='b', marker='o')

                else: # use m input to optimize there other parameters.
                    para = (a,b,g,m,0) # alpha, beta, gamma, m, phi
                    ets = ets_model(state0=state0, parameters=para)
                    #for i in range(0,len(y)):
                    for i in range(1,len(y)):
                        ets.predict_observe(y[i])

                    # store performance coressponding to the alpha n beta
                    if (ets.sse < sse):
                        sse = ets.sse
                        alpha = a
                        beta  = b
                        gamma = g
                        season = m
                        #print("fixed best ets_abg a={} b={} g={} m={} sse={}".format(a, b, g, m, sse))
                    #elif(a%0.2 == 0) and (b%0.1 == 0) and (g%0.3 == 0):
                    #    print("fixed ets_abg a={} b={} g={} m={} sse={}".format(a, b, g, m, ets.sse))

                    # plot
                    #ax.scatter(xs=a, ys=g, zs=ets.sse, c='b', marker='o')

    #ax.set_xlabel('Alpha')
    #ax.set_ylabel('Gamma')
    #ax.set_zlabel('SSE')
    #print("alpha={} beta={} gamma={} m={}".format(alpha, beta, gamma, season))
          # alpha, beta, gamma,      m, phi
    return (alpha, beta, gamma, season,   0) 


##############################################################################
# optimizer, find parameters (alpha,beta,gamma,m,phi) to minimize SSE
def optimize_model(y, model, name, state0, m=1):
    '''
    y     : the dataset w/o time index
    model : ETS model to be optimized.
    name  : string name of the model.
    state0: the initial state to begin optimization
    m     : the period of the seasonality, e.g. m=4 for quarterly, 12 for monthly
    '''

    # ETS models - find & return the opimimal parameters
    model_ets_a   = ['ANN','MNN']             # alpha only
    model_ets_ab  = ['AAN','AMN','MAN','MMN'] # alpha, beta
    model_ets_ag  = ['ANA','ANM','MNA','MNM'] # alpha, gamma
    model_ets_abp = ['AAdN']                  # alpha, beta, phi
    model_ets_abg = ['AAA','AAM','AMA','AMM','MAA','MAM','MMA','MMM'] # alpha, beta, gamma

    if (name in model_ets_a):
        return optimizer_ets_a(y, model, state0)
    elif (name in model_ets_ab):
        return optimizer_ets_ab(y, model, state0)
    elif (name in model_ets_ag):
        return optimizer_ets_ag(y, model, state0, m)
    elif (name in model_ets_abp):
        return optimizer_ets_abp(y, model, state0)
    elif (name in model_ets_abg):
        return optimizer_ets_abg(y, model, state0, m)
    else: #default benchmark model
        return [] # naive model has no parameter


