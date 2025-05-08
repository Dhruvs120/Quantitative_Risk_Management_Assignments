#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
libGAS.py

Purpose:
    Estimate data from a gas(1,1) model

Version:
    1       First start, following simgas.py
    2       Copy from models/gas/gas.py, adapted for GARCH and use with qfrm

Note:
    The GAS(1,1) corresponds to the GARCH(1,1), with a slight change of parameters.

    If we have
        GARCH:
          S2(t+1)= omega + alpha a(t)^2 + beta S2(t)
        GAS:
          S2(t+1)= O + A s(t) + B S2(t)
    with s(t) the score in the GAS filter, then we have correspondence if
      O = omega
      A = alpha
      B = alpha + beta

Date:
    2019/6/21

Author:
    Charles Bos
"""
###########################################################
### Imports
# cd ..
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.dates as mdates

import os
import datetime as dt

###########################################################
### Get hessian and related functions
from grad import *
from readarg import *

###########################################################
### (vY, vS2)= GenrGAS(vP, iT)
def InitialiseGAS(dtArg):
    """
    Purpose:
        Initialise settings, prepare data

    Inputs:
        vP      vector of size 3, with dO, dA, dB
        iT      integer, number of observations

    Return value:
        vY      iT vector, observations
        vS2     iT vector, variances
    """
    ReadArg(dtArg)

    np.random.seed(dtArg['seed'])

    vP0= dtArg['p0']
    iT= dtArg['t']
    (vY, vS20)= GenrGAS(vP0, iT)
    vS2= np.zeros(iT)
    FiltGAS(vS2, vY, vP0)
    print ('Are they again equal?', np.all(vS2 == vS20))

    vL= np.exp(vY/100).cumsum()
    dtD1= dt.datetime.today().date()
    vD= pd.date_range(dtD1- dt.timedelta(days= iT-1), dtD1)

    df= pd.DataFrame(vL, columns= ['y'], index= vD)
    df['Return']= vY
    df['s20']= vS20

    dtArg['in']= f'data/sim_{df.index[0].year}_{df.index[-1].year}.csv.gz'

    if (len(dtArg['d1'])):
        vI= df.index >= dtArg['d1']
        if (vI.sum() > 100):
            df= df[vI]

    dtArg['data.df']= df
    dtArg['y']= df['Return'].dropna().values
    dtArg['t']= pd.to_datetime(df['Return'].dropna().index)

    sBase= os.path.basename(dtArg['in'])
    sSt= sBase.split('_')[0]

    iY0= dtArg['t'][0].year
    iY1= dtArg['t'][-1].year
    dtArg['out']= f'graphs/{sSt}_{iY0}_{iY1}_vol.png'

    return dtArg['y']

###########################################################
### (vY, vS2)= GenrGAS(vP, iT)
def GenrGAS(vP, iT):
    """
    Purpose:
        Generate GAS(1,1) data

    Inputs:
        vP      vector of size 3, with dO, dA, dB
        iT      integer, number of observations

    Return value:
        vY      iT vector, observations
        vS2     iT vector, variances
    """
    (dO, dA, dB)= (vP[0], vP[1], vP[2])
    vY= np.zeros(iT)
    vS2= np.zeros_like(vY)
    dF= dO/(1-dB)
    for i in range(iT):
        vY[i]= np.sqrt(dF) * np.random.randn()
        vS2[i]= dF
        dF= dO + dA*(vY[i]**2 - dF) + dB*dF

    return (vY, vS2)

###########################################################
### vPTr= TransPar(vP)
def TransPar(vP):
    """
    Purpose:
      Transform the parameters for restrictions

    Inputs:
      vP        array of size 3, with parameters O, A, B

    Return value:
      vPTr      array of size 3, with transformed O, A, B
    """
    vPTr= np.copy(vP)
    vPTr[0]= np.log(vP[0])
    vPTr[1:]= np.log(vP[1:]/(1-vP[1:]))

    return vPTr

###########################################################
### vP= TransBackPar(vPTr)
def TransBackPar(vPTr):
    """
    Purpose:
      Transform the parameters back from restrictions

    Inputs:
      vPTr      array of size 3, with transformed O, A, B

    Return value:
      vP        array of size 3, with parameters O, A, B
    """
    vP= np.copy(vPTr)
    vP[0]= np.exp(vPTr[0])
    vP[1:]= np.exp(vPTr[1:])/(1+np.exp(vPTr[1:]))

    return vP

###########################################################
### vPTr= TransPar(vP)
def TransParL(dL):
    """
    Purpose:
      Transform the parameters for restrictions

    Inputs:
      dLambda   double

    Return value:
      dLTr      double, transformed Lambda
    """
    dLTr= np.log(dL/(1-dL))

    return dLTr

###########################################################
### vP= TransBackPar(vPTr)
def TransBackParL(dLTr):
    """
    Purpose:
      Transform the lambda parameters back from restrictions

    Inputs:
      dLTr      double, transformed Lambda

    Return value:
      dLambda   double
    """
    dL= np.exp(dLTr)/(1+np.exp(dLTr))

    return dL

###########################################################
### br= FiltGAS(vS2, vY, vP)
def FiltGAS(vS2, vY, vP):
    """
    Purpose:
        Filter the process using the GAS equations

    Inputs:
        vY      iT vector, observations
        vP      vector of size 3, with dO, dA, dB
        vS2     iT vector, empty space

    Outputs:
        vS2     iT vector, variances

    Return value:
        br      boolean, True if all went well
    """
    iT= vY.shape[0]
    (dO, dA, dB)= (vP[0], vP[1], vP[2])

    dF= dO/(1-dB)
    for i in range(iT):
        vS2[i]= dF

        # ### Check
        # dNabla= -0.5*(1-(vY[i])**2/dF)/dF;
        # dI= 0.5/(dF**2)
        #
        # dS= 1.0 / dI
        # ds= dS * dNabla
        # if (np.fabs(ds - ((vY[i])**2 - dF)) > 1e-5):
        #     print ('i=%i s= %g, salt= %g: ', i, ds, ((vY[i]**2) - dF))

        ds= (vY[i])**2 - dF
        dF= dO + dA * ds + dB * dF

    return not np.any(np.isnan(vS2))

###########################################################
### vLL= LnLGAS(vP, vY)
def LnLGAS(vP, vY):
    """
    Purpose:
        Calculate vector of LL using the GAS equations

    Inputs:
        vP      vector of size 3, with dO, dA, dB
        vY      iT vector, observations


    Return value:
        vLL     iT vector, loglikelihoods
    """
    iT= vY.shape[0]
    vS2= np.zeros_like(vY)

    br= FiltGAS(vS2, vY, vP)
    vLL= st.norm.logpdf(vY/np.sqrt(vS2)) - 0.5*np.log(vS2)

    # vLL0= -0.5*(np.log(2*np.pi) + np.log(vS2) + (vY**2)/vS2)
    # dDiff= np.max(np.abs(vLL - vLL0))
    # print ('ll= %g, diff=%g, o=%g, a=%g, b=%g' % (vLL.mean(), dDiff, vP[0], vP[1], vP[2]))
    # print ('ll= %g, o=%g, a=%g, b=%g' % (vLL.mean(), vP[0], vP[1], vP[2]))

    # vPTr= TransPar(vP)
    # print ('oTr=%g, aTr= %g, bTr= %g' % (vPTr[0], vPTr[1], vPTr[2]))

    return vLL

###########################################################
### (dLL, vS2, sMess)= EstGAS(vP, vY)
def EstGAS(vP, vY):
    """
    Purpose:
        Optimise GAS model

    Inputs:
        vP      vector of size 3, with dO, dA, dB, starting values
        vY      iT vector, observations

    Outputs:
        vP      3-vector, optimised parameters

    Return value:
        dLL     double, optimal loglikelihood
        vS2     iT vector, time varying variance
        sMess   string, message on convergence
    """
    iT= vY.shape[0]

    # Create function returning NEGATIVE average LL, as function of vP
    AvgNLnLGAS= lambda vP: -(LnLGAS(vP, vY).mean())

    vP0= np.copy(vP)
    res= opt.minimize(AvgNLnLGAS, vP0, method='BFGS')

    vP[:]= res.x
    sMess= res.message
    dLL= -iT*res.fun

    print ('\nBFGS results in ', sMess, '\nPars: ', vP, '\nLL= ', dLL, ', f-eval= ', res.nfev)

    return (dLL, vS2, sMess)

###########################################################
### (dLL, vS2, sMess)= EstGAS(vP, vY)
def EstGASTr(vP, vY):
    """
    Purpose:
        Optimise GAS model, using transformation

    Inputs:
        vP      vector of size 3, with dO, dA, dB, starting values
        vY      iT vector, observations

    Outputs:
        vP      3-vector, optimised parameters

    Return value:
        dLL     double, optimal loglikelihood
        vS2     iT vector, time varying variance
        sMess   string, message on convergence
    """
    iT= vY.shape[0]

    # Create function returning NEGATIVE average LL, as function of vP
    AvgNLnLGASTr= lambda vPTr: -(LnLGAS(TransBackPar(vPTr), vY).mean())

    vPTr= TransPar(vP)
    dLL= -iT*AvgNLnLGASTr(vPTr)
    print ('Initial LL=%g' % dLL)

    res= opt.minimize(AvgNLnLGASTr, vPTr, method='BFGS')

    vP[:]= TransBackPar(res.x)
    sMess= res.message
    dLL= -iT*res.fun
    vS2= np.zeros(iT)
    FiltGAS(vS2, vY, vP)

    print ('\nBFGS results in ', sMess, '\nPars: ', vP, '\nLL= ', dLL, ', f-eval= ', res.nfev)

    return (dLL, vS2, sMess)



###########################################################
### br= FiltEWMA(vY, dLambda)
def FiltEWMA(vY, dLambda, s20= None):
    """
    Purpose:
        Apply EWMA filter of variance

    Inputs:
        vY          iT vector, observations
        dLambda     double, value for weight
        s20         (optional, default is overall variance) double, initial variance

    Return value:
        vS2     iT vector, time varying variance
    """
    # dLambda= 0.8
    s20= s20 or np.var(vY)

    iT= len(vY)
    vS2= np.zeros_like(vY)
    vS2[0]= s20
    for i in range(0, iT):
        vS2[i]= s20
        s20= dLambda * s20 + (1-dLambda)*(vY[i]**2)

    return vS2



###########################################################
### dD2= AvgEWMAdist(vY, dLambda)
def AvgEWMAdist(vY, dLambda, s20= None):
    """
    Purpose:
        Calculate average misfit of EWMA filter of variance

    Inputs:
        vY          iT vector, observations
        dLambda     double, value for weight

    Return value:
        dD2         double, average difference between vY^2 and vS2
    """
    iT= len(vY)
    vS2= FiltEWMA(vY, dLambda, s20= s20)

    vD= (vY**2) - vS2
    dD2= np.mean(vD**2)

    print (f'l {dLambda}, s20= {s20}, d= {dD2}')

    return dD2

###########################################################
### main
def EstEWMATr(vLambda, vY):
    """
    Purpose:
        Estimate an EWMA filter of variance

    Inputs:
        vY          iT vector, observations
        dLambda0    (optional, default= 0.8) initial value for weight

    Return value:
        dLambda double, optimal weight
        vS2     iT vector, time varying variance
    """
    iT= len(vY)

    # Create function returning average distance to minimize
    AvgEWMAdistLTr= lambda vLTr: np.mean((vY**2 - FiltEWMA(vY, TransBackParL(vLTr)[0]))**2)
    # AvgEWMAdistLTr= lambda vLTr: AvgEWMAdist(vY, TransBackParL(vLTr)[0])

    dLambda0= vLambda[0]
    vLTr= np.array([TransParL(dLambda0)])

    dD= iT*AvgEWMAdistLTr(vLTr)
    print (f'Initial D={dD}')

    res= opt.minimize(AvgEWMAdistLTr, vLTr, method='BFGS')
    dLambda= TransBackParL(res.x)[0]
    sMess= res.message
    dD= iT*res.fun

    vS2= FiltEWMA(vY, dLambda)
    vLambda[0]= dLambda

    print ('\nBFGS results in ', sMess, '\nPars: ', dLambda, '\nD= ', dD, ', f-eval= ', res.nfev)

    return (dD, vS2, sMess)


###########################################################
### main
def main():
    # Magic numbers
    dtArg= { 'in': '',
             'd1': '2010-01-01',
             't': 1000,
             'seed': 1234,
             'p0': [.1, .05, .95],
             'l0': .8,
             'lRM': .94,
           }

    # Initialisation
    vP0= dtArg['p0']
    vY= InitialiseGAS(dtArg)
    vT= dtArg['t']
    sOut= dtArg['out']

    # Estimation
    # vP= np.copy(vP0)
    # (dLL, sMess)= EstGAS(vP, vY)

    vP= np.copy(vP0)
    (dLL, vS2, sMess)= EstGASTr(vP, vY)

    vL0= np.array([dtArg['l0']])
    (dD, vS2e, sMesse)= EstEWMATr(vL0, vY)
    vS2rm= FiltEWMA(vY, dtArg['lRM'])

    # Output

    plt.figure(figsize= (8, 4))
    ax= plt.subplot(1, 2, 1)
    plt.plot(vT, vY, 'b.', label='y')
    if ('s20' in dtArg):
        plt.plot(vT, -np.sqrt(dtArg['s20']), 'g-', label='Generated sdev')
    plt.plot(vT, 2*np.sqrt(vS2), 'r-', label='Volatility GARCH')
    plt.plot(vT, 2*np.sqrt(vS2e), 'g-', label=f'Volatility EWMA(l={vL0[0]:.2f})')
    plt.plot(vT, 2*np.sqrt(vS2rm), 'y-', label=f'Volatility EWMA(lRM={dtArg["lRM"]})')

    plt.plot(vT, -2*np.sqrt(vS2), 'r-')
    plt.plot(vT, -2*np.sqrt(vS2e), 'g-')
    plt.plot(vT, -2*np.sqrt(vS2rm), 'y-')

    fmTime = mdates.DateFormatter('%y')
    ax.xaxis.set_major_formatter(fmTime)

    plt.legend()

    ax= plt.subplot(1, 2, 2)
    plt.plot(vT, np.sqrt(vS2), 'r-', label='Volatility GARCH')
    plt.plot(vT, np.sqrt(vS2e), 'g-', label=f'Volatility EWMA(l={vL0[0]:.2f})')
    plt.plot(vT, np.sqrt(vS2rm), 'y-', label=f'Volatility EWMA(lRM={dtArg["lRM"]})')
    ax.xaxis.set_major_formatter(fmTime)

    # plt.legend()
    plt.savefig(dtArg['out'])
    plt.show()

###########################################################
### start main
if __name__ == '__main__':
    main()
