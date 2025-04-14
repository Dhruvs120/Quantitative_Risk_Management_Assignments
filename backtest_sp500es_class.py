#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_sp500.py

Purpose:
    Estimate data from a garch(1,1) model, then do a backtest

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
from estgarch_sp500 import *

###########################################################
### main
def PrepModel(dtArg):
    df= dtArg['data.df']

    dtG= EstGARCH(df['Return'].values)

    dtArg['pars.df']= dtG['pars']

    df['s2GARCH']= dtG['s2']
    df['Loss']= -df['Return']

###########################################################
### dVaR= VaR(alpha, r, s)
def VaR(alpha, r= 0, s= 1, df= 0):
    """
    Purpose:
        Get the VaR of the normal model

    Inputs:
        alpha   double, level
        r       double, expected return
        s       double, volatility
        df      (optional) double, degrees of freedom for student-t

    Return value:
        dVaR    double, VaR
    """
    if (df == 0):
        dVaR0= st.norm.ppf(alpha)

        dVaR= r + s*dVaR0
    else:
        dVaR0= st.t.ppf(alpha, df= df)

        dS2t= df/(df-2)
        c= s / np.sqrt(dS2t)
        dVaR= r + c*dVaR0

    return dVaR

###########################################################
### dES= ES(alpha, r, s)
def ES(alpha, r= 0, s= 1, df= 0):
    """
    Purpose:
        Get the ES of the normal/student model

    Inputs:
        alpha   double, level
        r       double, expected return
        s       double, volatility
        df      (optional, default= 0/normal) double, df

    Return value:
        dES     double, ES
    """
    if (df == 0):
        dVaR0= st.norm.ppf(alpha)
        dES0= st.norm.pdf(dVaR0) / (1-alpha)
        dES= r + s*dES0
    else:
        dVaR0= st.t.ppf(alpha, df= df)
        dES0= st.t.pdf(dVaR0, df= df)*((df + dVaR0**2)/(df-1)) / (1-alpha)

        dS2t= df/(df-2)
        c= s / np.sqrt(dS2t)
        dES= r + c*dES0

    return dES

###########################################################
### main
def PrepVaR(dtArg):
    df= dtArg['data.df']
    vAlpha= dtArg['alpha']
    dfPars= dtArg['pars.df']
    vY= df['Return']

    dMu= dfPars.loc['mu', 'p']

    vS= np.sqrt(df['s2GARCH'])

    dfVaR= pd.DataFrame()
    for alpha in vAlpha:
        dfVaR[f'GARCH({alpha})']= VaR(alpha, r= dMu, s= vS, df= 0)

    dtArg['var.df']= dfVaR

###########################################################
### main
def PrepES(dtArg):
    df= dtArg['data.df']
    dfVaR= dtArg['var.df']
    vAlpha= dtArg['alpha']
    dfPars= dtArg['pars.df']

    vY= df['Return']
    vLoss= df['Loss']

    dMu= dfPars.loc['mu', 'p']
    vS= np.sqrt(df['s2GARCH'])

    dfES= pd.DataFrame()
    for alpha in vAlpha:
        dfES[f'GARCH({alpha})']= ES(alpha, r= dMu, s= vS, df= 0)

    dtArg['es.df']= dfES

###########################################################
### main
def PrepViol(dtArg):
    df= dtArg['data.df']
    dfVaR= dtArg['var.df']

    dfViol= pd.DataFrame()
    for c in dfVaR.columns:
        dfViol[c]= df['Loss'] > dfVaR[c]

    dtArg['viol.df']= dfViol

###########################################################
### TestTES(srLoss, dfViol, dfES)
def TestTES(srLoss, dfViol, dfES):
    """
    Purpose:
        Perform a t-test on the violation residuals

    Inputs:
        srLoss  series, loss over time
        dfViol  dataframe, boolean indicators if loss is larger than VaR
        dfES    dataframe, expected shortfalls

    Return value:
        dfTES   dataframe, t-value and p-value of t-test for each column of dfES
        dfK     dataframe, violation residuals
    """
    (iN, iK)= dfViol.shape
    dMu= srLoss.mean()
    asC= dfES.columns
    dfTES= pd.DataFrame(columns= asC)

    # One could compute all violation residuals at once
    # Use conditional assignment: Only assign residual when violation took place
    # mK= np.where(dfViol, (srLoss.values.reshape(-1, 1) - dfES) / (dfES - dMu), 0)
    # dfK= pd.DataFrame(mK, index= dfES.index, columns= dfES.columns)

    dfK= pd.DataFrame(index= dfES.index, columns= asC)
    for (c, sC) in enumerate(asC):
        # Use conditional assignment: Only assign residual when violation took place
        dfK[sC]= np.where(dfViol.iloc[:, c], (srLoss - dfES[sC]) / (dfES[sC] - dMu), 0)

        vI= dfViol[sC]
        vKv= dfK.loc[vI, sC]

        iDf= len(vKv)
        dS2k= vKv.var()
        dS2kavg= dS2k/iDf
        dSv= np.sqrt(dS2kavg)
        dMv= vKv.mean()
        dT= (dMv - 0)/dSv
        dPt0= st.t.cdf(dT, df= iDf)
        dPt= 2*min(dPt0, 1-dPt0)

        dfTES.loc['K-mean', sC]= dMv
        dfTES.loc['K-sdev', sC]= dSv
        dfTES.loc['t-ES', sC]= dT
        dfTES.loc['p-ES', sC]= dPt

    return (dfTES, dfK)


###########################################################
### main
def main():
    # Magic numbers
    dtArg= {
             'in': 'data/sp500_1995_2025.csv.gz',
             'alpha': [.95, .99],
             # 'alphatest': .05,
             # 'lambda': .94,
           }

    # Initialisation
    # dAlphaTest= dtArg['alphatest']= .95
    vY= Initialise(dtArg)
    vAlpha= dtArg['alpha']

    # Estimation
    PrepModel(dtArg)

    PrepVaR(dtArg)
    PrepES(dtArg)
    PrepViol(dtArg)

    df= dtArg['data.df']
    srLoss= df['Loss']
    dfViol= dtArg['viol.df']
    dfES= dtArg['es.df']

    # Output
    (dfTES, dfK)= TestTES(srLoss, dfViol, dfES)
    print ('t-test results for Expected Shortfall:')
    print (dfTES)

    # Show shortfall residual
    iC= dfK.shape[1]
    plt.figure()
    for (i, sC) in enumerate(dfK.columns):
        plt.subplot(1, iC, i+1)
        vI= dfViol[sC]
        dMk= dfK.loc[vI, sC].mean()
        plt.hist(dfK.loc[vI, sC], label= f'm= {dfTES.loc["K-mean", sC]:.2f}, t= {dfTES.loc["t-ES", sC]:.2f}')
        plt.legend()
        plt.title(f'Shortfall residual (a= {vAlpha[i]})')
    plt.show()


###########################################################
### start main
if __name__ == '__main__':
    main()
