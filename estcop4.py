#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EstCOP.py

Purpose:
    Estimate copula

Version:
    1       First start, based on l2/estgas.py
    2       Include Gumbel/Clayton copula estimation
    3       Trying to get Gumbel working right...
    4       Cleaning out checking code

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
    2019/6/21, 2025/3/20

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
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula, ClaytonCopula, IndependenceCopula

###########################################################
### Get hessian and related functions
from lib.grad import *
from lib.readarg import *

from lib.libgas import *

###########################################################
### (vY, vS2)= GenrGAS(vP, iT)
def Initialise(dtArg):
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

    df= pd.read_csv(dtArg['in'], parse_dates= ['Date'], index_col= 'Date')
    asC= df.columns
    for c in asC:
        df[f'R{c}']= 100*np.log(df[c]).diff()
    dtArg['vars']= asVars= [ f'R{c}' for c in asC ]
    dtArg['tickers']= asC

    if (len(dtArg['d1'])):
        vI= df.index >= dtArg['d1']
        if (vI.sum() > 100):
            df= df[vI]

    # Drop missings
    df= df.dropna().copy()

    # Get rid of mean return
    vM= df[asVars].mean()
    df[asVars]-= vM

    dtArg['data.df']= df
    dtArg['y']= df[asVars]

    return df

###########################################################
### main
def EstimMargin(dtArg):
    """
    Purpose:
        Estimate GARCH-Normal margins on the variables of interest

    Inputs:
        dtArg   dictionary, settings with data.df, vars

    Outputs:
        dtArg   dictionary, with res.df with parameter estimates, and data.df updated with S2 and U
    """
    vP0= dtArg['p0']
    df= dtArg['data.df']
    avP= []
    for r in dtArg['vars']:
        vP= np.copy(vP0)
        vY= df[r].values
        dM= vY.mean()
        print (f'\n=====\nEstimating model for "{r}"')
        (dLL, vS2, sMess)= EstGASTr(vP, vY-dM)
        df[ 'S2' + r[1:]]= vS2
        vU= st.norm.cdf((vY-dM) / np.sqrt(vS2))
        df[ 'U' + r[1:]]= vU

        avP.append(vP)
    dtArg['res.df']= pd.DataFrame(avP, index= dtArg['vars'], columns= ['O', 'A', 'B'])

###########################################################
### main
def DisplayDF(dtArg):
    """
    Purpose:
        Display the output of the margin estimation
    """
    df= dtArg['data.df']
    vT= df.index
    asTick= dtArg['tickers']
    asUTick= 'U' + asTick

    fmTime = mdates.DateFormatter('%y')
    sOut= dtArg['in'].replace('data', 'graphs').replace('.csv.gz', 'cop.png')

    plt.figure(figsize= (8, 4))
    ax= plt.subplot(2, 3, 1)

    plt.plot(df[asTick], '-', label=asTick)
    plt.legend()
    ax.xaxis.set_major_formatter(fmTime)

    ax= plt.subplot(2, 3, 4)
    plt.plot(df['U' + asTick[0]], df['U' + asTick[1]], '.', label= 'U(GARCH)')
    plt.legend()

    for i in range(2):
        ax= plt.subplot(2, 3, 2+3*i)
        plt.plot(df['R' + asTick[i]], '.', label= asTick[i])
        plt.plot(2*np.sqrt(df['S2'+asTick[i]]), 'r-')
        plt.plot(-2*np.sqrt(df['S2'+asTick[i]]), 'r-')
        plt.legend()
        ax.xaxis.set_major_formatter(fmTime)

    for i in range(2):
        ax= plt.subplot(2, 3, 3+3*i)
        plt.plot(df['U' + asTick[i]], '.', label= asTick[i])
        plt.legend()
        ax.xaxis.set_major_formatter(fmTime)

    plt.savefig(sOut)
    plt.show()

    sOut= dtArg['in'].replace('data', 'graphs').replace('.csv.gz', 'copu.png')
    plt.figure(figsize= (8, 4))
    plt.plot(df[asUTick[0]], df[asUTick[1]], '.')
    plt.xlabel(asTick[0])
    plt.ylabel(asTick[1])
    plt.savefig(sOut)
    plt.show()

###########################################################
### vPTr= TransPar(vP, mod= 'gauss')
def TransParC(vP, mod= 'gauss'):
    """
    Purpose:
      Transform the parameters for restrictions

    Inputs:
      vP        array of size 1 or 2, with parameters rho and nu, or theta

    Return value:
      vPTr      array, with transformed rho and nu, or theta
    """
    vPTr= np.copy(vP)
    if (mod in ['gauss', 'stud']):
        vPTr[0]= np.log(vP[0]/(1-vP[0]))
        if (len(vPTr) > 1):
            vPTr[1]= np.log(vP[1]-2)
    else:
        dLim= -1 if mod.startswith('clayton') else 1
        vPTr[0]= np.log(vPTr[0] - dLim)

    return vPTr

###########################################################
### vP= TransBackPar(vPTr)
def TransBackParC(vPTr, mod= 'gauss'):
    """
    Purpose:
      Transform the parameters back from restrictions

    Inputs:
      vPTr      array, with transformed rho and nu, or theta

    Return value:
      vP        array of size 1 or 2, with parameters rho and nu, or theta
    """
    vP= np.copy(vPTr)
    if (mod in ['gauss', 'stud']):
        vP[0]= np.exp(vPTr[0])/(1+np.exp(vPTr[0]))
        if (len(vP) > 1):
            vP[1]= 2+np.exp(vPTr[1])
    else:
        dLim= -1 if mod.startswith('clayton') else 1
        vP[0]= dLim+np.exp(vPTr[0])

    return vP

###########################################################
### vLL= LnLCopExpl(vP, mU, mod= 'clayton')
def LnLCopExplSM(vP, mU, mod= 'gumbelSM'):
    """
    Purpose:
        Calculate the loglikelihood according to the explicit copula density, through statsmodels
    """
    # vP= TransBackParC(vPTr, mod= mod)
    iT= mU.shape[0]
    dTheta= vP[0]

    if (mod == 'gumbelSM'):
        copula= GumbelCopula(theta= dTheta)
    else:
        copula= ClaytonCopula(theta= dTheta)
    vLL= copula.logpdf(mU)

    return vLL

###########################################################
### vLL= LnLCopExpl(vP, mU, mod= 'clayton')
def LnLCopExpl(vP, mU, mod= 'gumbel'):
    """
    Purpose:
        Calculate the loglikelihood according to the explicit copula density
    """
    # vP= TransBackParC(vPTr, mod= mod)
    iT= mU.shape[0]
    dTheta= vP[0]
    if (mod == 'gumbel'):
        # Generator
        fnPsi= lambda t: np.exp(-t**(1/dTheta))
        # First derivative of generator
        fnpsi= lambda t: -t**(1/dTheta-1) * fnPsi(t) / dTheta
        # Second derivative of generator
        fnpsi2= lambda t: t**(1/dTheta-2) * (-1 + dTheta + t**(1/dTheta)) * fnPsi(t) / dTheta**2

        # Inverse generator x(u)
        fnPsiI= lambda u: (-np.log(u))**dTheta
        # First derivative of inverse generator x(u)
        fnpsiI= lambda u: -dTheta*(-np.log(u))**(dTheta-1) / u
    elif (mod == 'clayton'):
        # Generator
        fnPsi= lambda t: (1+dTheta*t)**(-1/dTheta)
        # First derivative of generator
        fnpsi= lambda t: -(1+dTheta*t)**(-1/dTheta-1)
        # Second derivative of generator
        fnpsi2= lambda t: (1+dTheta)*(1+dTheta*t)**(-1/dTheta-2)

        # Inverse generator x(u)
        fnPsiI= lambda u: (u**(-dTheta) - 1)/dTheta
        # First derivative of inverse generator x(u)
        fnpsiI= lambda u: -u**(-dTheta-1)
    else:
        print (f'Error: Model {mod} not recognised')
        return np.nan

    # Get the loglikelihood
    mX= fnPsiI(mU)
    vX= mX.sum(axis= 1)
    vL= fnpsi2(vX)*fnpsiI(mU).prod(axis= 1)

    vLL= np.log(vL)

    print (f'l: {vLL.sum():.4f}, th: {dTheta:.4f}')
    if (len(vLL) != iT):
        print (f'Error: shape of vLL= {vLL.shape}, should be ({iT},)')

    return vLL

###########################################################
### vLL= LnLCopImpl(vP, mU, mod= 'stud')
def LnLCopImpl(vP, mU, mod= 'gauss'):
    """
    Purpose:
        Calculate vector of LL using the Copula equations

    Inputs:
        vP      vector of size 1 or 2, with dRho, dNu
        mU      iT x 2 matrix, observations


    Return value:
        vLL     iT vector, loglikelihoods
    """
    if (mod == 'gauss'):
        dNu= np.inf
        fnf= st.norm()
    elif (mod == 'stud'):
        dNu= vP[1]
        fnf= st.t(dNu)
    else:
        print (f'Error: Model {mod} not recognised')
        return np.nan

    iT= mU.shape[0]
    dRho= vP[0]
    mP= np.array([[1, dRho], [dRho, 1]])
    mC= np.linalg.cholesky(mP)
    mCi= np.linalg.inv(mC)
    (iS, dLogD)= np.linalg.slogdet(mP)

    mX= fnf.ppf(mU)
    vF= fnf.logpdf(mCi@mX.T).sum(axis= 0) - 0.5*dLogD
    vG= fnf.logpdf(mX.T).sum(axis= 0)
    vLL= vF - vG

    if (len(vLL) != iT):
        print (f'Error: shape of vLL= {vLL.shape}, should be ({iT},)')

    return vLL

###########################################################
### vLL= LnLGAS(vP, mU)
def LnLCop(vP, mU, mod= 'gauss'):
    """
    Purpose:
        Calculate vector of LL using the Copula equations

    Inputs:
        vP      vector of size 1 or 2, with dRho, dNu, or theta
        mU      iT x 2 matrix, observations

    Return value:
        vLL     iT vector, loglikelihoods
    """
    if (mod in ['gumbel', 'clayton']):
        return LnLCopExpl(vP, mU, mod= mod)
    if (mod in ['gumbelSM', 'claytonSM']):
        return LnLCopExplSM(vP, mU, mod= mod)
    return LnLCopImpl(vP, mU, mod= mod)

###########################################################
### main
def EstCopula(dtArg, mod= 'gauss'):
    # mod= 'gumbel'
    df= dtArg['data.df']
    asTick= dtArg['tickers']
    asU= 'U' + asTick
    mU= df[asU]
    iT= mU.shape[0]

    # mR= mU.rank()
    mR= mU          # No need to take the ranks, scipy.stats will take those
    res= st.kendalltau(mR.iloc[:, 0], mR.iloc[:,1])
    dRtau= res.statistic
    if (mod in ['gauss', 'stud']):
        mP0= mU.corr()
        dP0= mP0.iloc[0, 1]
        dNu= 4.0
    elif (mod.startswith('gumbel')):
        dTheta= dP0= 1/(1-dRtau)
    else:
        res= st.spearmanr(mU)
        dRsp= res.statistic
        dTheta= dP0= dRsp
        dP0= 2.0
        dTheta= dP0= 2*dRtau/(1-dRtau)
    vP0= np.array([dP0, dNu]) if mod == 'stud' else np.array([dP0])

    # Create function returning NEGATIVE average LL, as function of vP
    AvgNLnLCopTr= lambda vPTr: -(LnLCop(TransBackParC(vPTr, mod= mod), mU, mod= mod).mean())

    vPTr= TransParC(vP0, mod= mod)
    dALnL= AvgNLnLCopTr(vPTr)
    res= opt.minimize(AvgNLnLCopTr, vPTr, method='BFGS')

    vP= TransBackParC(res.x, mod= mod)
    sMess= res.message
    dLL= -iT*res.fun

    print ('\nBFGS results in ', sMess, '\nPars: ', vP, '\nLL= ', dLL, ', f-eval= ', res.nfev)

    # LnLCopL= lambda x: -(LnLCop([x], mU, mod= mod).mean())
    # vT= np.arange(1, 3, .01)
    # vL= [ LnLCopL(t) for t in vT ]
    return (dLL, vP, sMess)

###########################################################
### main
def main():
    # Magic numbers
    dtArg= { 'in': 'data/sie_bmw_1025_yf.csv.gz',
             'd1': '2010-01-01',
             'p0': [.1, .05, .95],
             'l0': .8,
             'lRM': .94,
           }

    # Initialisation
    df= Initialise(dtArg)
    dtArg.keys()

    # Estimation
    EstimMargin(dtArg)

    asTick= dtArg['tickers']
    asU= 'U' + asTick
    mU= df[asU]
    mP0= mU.corr()
    dRho0= mP0.iloc[0, 1]

    # mR= mU.rank()
    mR= mU          # No need to take the ranks, scipy.stats will take those
    res= st.kendalltau(mR.iloc[:, 0], mR.iloc[:,1])
    dRtau= res.statistic
    dThetaGu= 1/(1-dRtau)
    dThetaCl= 2*dRtau/(1-dRtau)

    dfR= pd.DataFrame(index= ['Rho', 'Nu', 'Theta', 'LL'])
    dfR.loc['Rho', 'Corr']= dRho0

    (dLLn, vPn, sMessn)= EstCopula(dtArg, mod= 'gauss')
    dfR.loc['Rho', 'Norm']= vPn
    dfR.loc['LL', 'Norm']= dLLn

    (dLLt, vPt, sMesst)= EstCopula(dtArg, mod= 'stud')
    dfR.loc[['Rho', 'Nu'], 't']= vPt
    dfR.loc['LL', 't']= dLLt

    dfR.loc['Theta', 'GuTau']= dThetaGu
    (dLLg, vPg, sMessg)= EstCopula(dtArg, mod= 'gumbel')
    dfR.loc['Theta', 'gumbel']= vPg
    dfR.loc['LL', 'gumbel']= dLLg

    (dLLg, vPg, sMessg)= EstCopula(dtArg, mod= 'gumbelSM')
    dfR.loc['Theta', 'gumbelSM']= vPg
    dfR.loc['LL', 'gumbelSM']= dLLg

    dfR.loc['Theta', 'ClTau']= dThetaCl
    (dLLc, vPc, sMessc)= EstCopula(dtArg, mod= 'clayton')
    dfR.loc['Theta', 'clayton']= vPc
    dfR.loc['LL', 'clayton']= dLLc

    (dLLc, vPc, sMessc)= EstCopula(dtArg, mod= 'claytonSM')
    dfR.loc['Theta', 'claytonSM']= vPc
    dfR.loc['LL', 'claytonSM']= dLLc

    # Output
    DisplayDF(dtArg)

    print ('Estimation results:')
    print (dfR.iloc[:, :3].dropna(axis= 0, how= 'all').to_latex(float_format= '%.3f'))
    print (dfR.iloc[:, 3:].dropna(axis= 0, how= 'all').to_latex(float_format= '%.3f'))

###########################################################
### start main
if __name__ == '__main__':
    main()
