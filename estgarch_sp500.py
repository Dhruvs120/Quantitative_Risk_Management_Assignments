#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EstGARCH_sp500.py

Purpose:
    Estimate data from a garch(1,1) model

Version:
    1       First start, following simgas.py
    2       Copy from models/gas/gas.py, adapted for GARCH and use with qfrm
    3       Pure GARCH, simplified

Note:
    We have
        GARCH:
          s2(t+1)= omega + alpha a(t)^2 + beta s2(t)
          a(t)= y(t) - mu
          a(t) ~ N(mu, s2(t))
Date:
    2019/6/21, 2025/4/2

Author:
    Charles Bos
"""
###########################################################
### Imports
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
from lib.grad import *

###########################################################
### vY= Initialise(dtArg)
def Initialise(dtArg):
    """
    Purpose:
        Initialise settings, prepare data

    Inputs:
        dtArg   dictionary, initial settings

    Outputs:
        dtArg['data.df']    dataframe, data including 'Return' column

    Return value:
        vY      iT vector, observations
    """
    df= pd.read_csv(dtArg['in'], index_col= 'Date')
    sC= df.columns[0]
    df['Return']= 100*np.log(df[sC]).diff()
    df= df.dropna()
    dtArg['data.df']= df

    vY= df['Return'].values

    return vY

###########################################################
### vPTr= TransPar(vP)
def TransPar(vP):
    """
    Purpose:
      Transform the parameters for restrictions

    Inputs:
      vP        array of size 4, with parameters mu, O, A, B

    Return value:
      vPTr      array of size 4, with transformed mu, O, A, B
    """
    vPTr= np.copy(vP)
    vPTr[1]= np.log(vP[1])
    vPTr[2:]= np.log(vP[2:]/(1-vP[2:]))

    return vPTr

###########################################################
### vP= TransBackPar(vPTr)
def TransBackPar(vPTr):
    """
    Purpose:
      Transform the parameters back from restrictions

    Inputs:
      vPTr      array of size 4, with transformed mu, O, A, B

    Return value:
      vP        array of size 4, with parameters mu, O, A, B
    """
    vP= np.copy(vPTr)
    vP[1]= np.exp(vPTr[1])                              # Restrict O > 0
    vP[2:]= np.exp(vPTr[2:])/(1+np.exp(vPTr[2:]))       # Restrict 0 < A,B < 1

    return vP

###########################################################
### vS2= FiltGARCH(vP, vY)
def FiltGARCH(vP, vY):
    """
    Purpose:
        Filter the process using the GARCH equations

    Inputs:
        vY      iT vector, observations
        vP      vector of size 4, with dMu, dO, dA, dB

    Return value:
        vS2     iT vector, variances
    """
    (dMu, dO, dA, dB)= (vP[0], vP[1], vP[2], vP[3])

    iT= vY.shape[0]
    vS2= np.zeros(iT)

    dS2= dO/(1-dA-dB)
    vA= vY - dMu
    for i in range(iT):
        vS2[i]= dS2

        dS2= dO + dA * vA[i]**2 + dB * dS2

    return vS2

###########################################################
### vLL= LnLGARCH(vP, vY)
def LnLGARCH(vP, vY):
    """
    Purpose:
        Calculate vector of LL using the GARCH equations

    Inputs:
        vP      vector of size 4, with dMu, dO, dA, dB
        vY      iT vector, observations

    Return value:
        vLL     iT vector, loglikelihoods
    """
    dMu= vP[0]
    vA= vY - dMu
    vS2= FiltGARCH(vP, vY)
    # vLL0= st.norm.logpdf((vY-dMu)/np.sqrt(vS2)) - 0.5*np.log(vS2)
    vLL= -0.5*(np.log(2*np.pi) + np.log(vS2) + (vA**2)/vS2)
    # dDiff= np.max(np.abs(vLL - vLL0))
    # print ('ll= %g, diff=%g, o=%g, a=%g, b=%g' % (vLL.mean(), dDiff, vP[0], vP[1], vP[2]))
    # print (f'll= {vLL.mean()}, m= {vP[0]}, o= {vP[1]}, a= {vP[2]}, b= {vP[3]}')

    return vLL

###########################################################
### dtG= EstGARCH(vY)
def EstGARCH(vY):
    """
    Purpose:
        Optimise GARCH model, using transformation

    Inputs:
        vY      iT vector, observations

    Return value:
      dtG   dictionary, with
        dLL     double, optimal loglikelihood
        dfR     dataframe, with p0, p, s
        vS2     iT vector, time varying variance
        sMess   string, message on convergence
    """
    iT= vY.shape[0]

    # Get some rough initial values
    vP0= np.array([0, 1, .05, .9])
    vP0[0]= vY.mean()
    vP0[1]= (1 - vP0[2] - vP0[3]) * np.var(vY)

    # Create function returning NEGATIVE average LL, as function of vP
    AvgNLnLGARCHTr= lambda vPTr: -(LnLGARCH(TransBackPar(vPTr), vY).mean())

    vPTr= TransPar(vP0)
    dLL= -iT*AvgNLnLGARCHTr(vPTr)
    print ('Initial LL=%g' % dLL)

    res= opt.minimize(AvgNLnLGARCHTr, vPTr, method='BFGS')

    vP= TransBackPar(res.x)
    sMess= res.message
    dLL= -iT*res.fun
    vS2= FiltGARCH(vP, vY)

    # Get standard errors, using delta method
    mA= -hessian_2sided(AvgNLnLGARCHTr, vPTr)
    mAi= np.linalg.inv(mA)
    mS2Tr= -mAi/iT
    mG= jacobian_2sided(TransBackPar, vPTr)  # Evaluate jacobian at vPTr
    mS2hd= mG @ mS2Tr @ mG.T                 # Cov(vP)
    vS= np.sqrt(np.diag(mS2hd))              # s(vP)

    dfR= pd.DataFrame(index= ['mu', 'omega', 'alpha', 'beta'])
    dfR['p0']= vP0
    dfR['p']= vP
    dfR['s']= vS

    print (f'\nBFGS results in LL= {dLL} (n-eval= {res.nfev}), with {sMess}\nParameters:\n', dfR)

    return {'ll': dLL, 'pars': dfR, 's2': vS2, 'message': sMess}


###########################################################
### DisplayS2(df)
def DisplayS2(df):
    vT= pd.to_datetime(df.index)
    [vY, vS2]= [ df[c].values for c in ['Return', 's2GARCH']]

    plt.figure(figsize= (8, 4))
    ax= plt.subplot(1, 2, 1)
    plt.plot(vT, vY, 'b.', label='y')
    plt.plot(vT, 2*np.sqrt(vS2), 'r-', label='Volatility GARCH')

    plt.plot(vT, -2*np.sqrt(vS2), 'r-')

    fmTime = mdates.DateFormatter('%y')
    ax.xaxis.set_major_formatter(fmTime)

    plt.legend()

    ax= plt.subplot(1, 2, 2)
    plt.plot(vT, np.sqrt(vS2), 'r-', label='Volatility GARCH')
    ax.xaxis.set_major_formatter(fmTime)
    # plt.legend()
    plt.show()

###########################################################
### main
def main():
    # Magic numbers
    dtArg= {
             'in': 'data/sp500_1995_2025.csv.gz',
           }

    # Initialisation
    vY= Initialise(dtArg)

    # Estimation
    dtG= EstGARCH(vY)

    # vL0= np.array([dtArg['l0']])
    # (dD, vS2e, sMesse)= EstEWMATr(vL0, vY)
    # vS2rm= FiltEWMA(vY, dtArg['lRM'])

    df= dtArg['data.df']
    df['s2GARCH']= dtG['s2']

    # Output
    DisplayS2(df)


###########################################################
### start main
if __name__ == '__main__':
    main()
