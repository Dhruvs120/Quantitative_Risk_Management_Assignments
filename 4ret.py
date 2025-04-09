#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4ret.py

Purpose:
    Show 4 returns

Version:
    1       First start, not used yet

Date:
    2025/4/2

Author:
    Charles Bos
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as st

###########################################################
### main
def main():
    # Magic numbers
    vR= [.15, .12, .10, .18]
    vW= [.25, .25, .25, .25]
    mS2= [[ 0.2, 0.06, 0.06, 0.05],
          [0.06, 0.16, 0.08, 0.1],
          [0.06, 0.08, 0.12, 0.07],
          [0.5, 0.1, 0.07, 0.22]]
    vAlpha= [.95, .975]

    # Initialisation
    vR= np.array(vR)
    vW= np.array(vW)
    mS2= np.array(mS2)


###########################################################
### start main
if __name__ == "__main__":
    main()
