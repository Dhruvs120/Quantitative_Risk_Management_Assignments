#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
readarg.py

Purpose:
    contain a series of routines for reading command line arguments

Version:
    1       Trying things out
    2       Allowing for a dictionary
    3       Allow for a dictionary argument 'args' which replaces the command line

Date:
    2018/12/10,

Author:
    Charles Bos
"""
###########################################################
### Imports
import numpy as np
import sys, re

#########################################################
### vRes= ReadArg('a', 5, vDef= [1, 2, 3])
### ir= ReadArg({'a': [1, 2, 3])
### val= ReadArg(2)
def ReadArg(sKey, iN= 0, vDef= None, show= False):
    """
    Purpose:
       Read the command line argument, returning a single value with whatever was read, or, when used with a dictionary, checking all the keys in the dictionary

    Inputs:
        sKey        string, label
        iN          integer, size and type indication. 0= boolean, 1= scalar, >1= vector, -1= string, <-1 = list of strings
        vDef        some type, default value
      or
        dtArg       dictionary, keys and default values

    Outputs:
        vRet        vector/list/scalar/boolean. Scalars are of type integer if possible, else float. If label is not found, the default value is returned
      or
        ir          integer, number of keys read from command line arguments
    """
    if (isinstance(sKey, dict)):
        # print ('Going to dict')
        return ReadArg_dict(sKey, show= show)

    lArg= sys.argv
    iA= len(lArg)
    iNa= np.fabs(iN)

    if (isinstance(sKey, int)):
        # val= ReadArg(3);      # Give third command line argument, or None if not found
        if (iA-1 > sKey):
            return lArg[sKey+1]
        return None

    if (iN == 0):
        vRet= (sKey in lArg)
        return vRet


    if (not sKey in lArg):
        return vDef
    j= lArg.index(sKey)+1
    vRet= []
    bCont= (j < iA)
    while bCont:
        if (iN > 0):
            try:
                iElement= float(lArg[j])
                vRet.append(iElement)
            except ValueError:
                print ("Arg '%s' cannot be converted to float" % lArg[j])
                bCont= False
        else:
            iElement= lArg[j]
            vRet.append(iElement)
        j+= 1
        bCont= bCont and (len(vRet) < iNa) and (j < iA)

    if (iN == 1):
        vRet= int(vRet[0]) if ((vRet[0] % 1) == 0) else vRet[0]
    elif (iN == -1):
        vRet= vRet[0]
    elif (iN > 1):
        vRet= np.array(vRet).astype(float)
    # print ("Key '", sKey, "': ", vRet, sep='')
    return vRet

#########################################################
def ReadArg_float(vArg):
    """
    Purpose:
        Read through the string elements in vArg, and transform them to floats if possible

    Inputs:
        vArg    array of length iA, string elements

    Return value:
        vRet    list, floats of elements of first iR elements which can be translated to floats
    """
    vRet= []
    i= 0
    iA= len(vArg)
    while (i < iA):
        try:
            dA= float(vArg[i])
            vRet.append(dA)
            i+= 1
        except ValueError:
            print ("Arg '%s' cannot be converted to float" % vArg[i])
            i= iA
    return vRet

#########################################################
def ReadArg_int(vArg):
    """
    Purpose:
        Read through the string elements in vArg, and transform them to ints if possible

    Inputs:
        vArg    array of length iA, string elements

    Return value:
        vRet    list, ints of elements of first iR elements which can be translated to ints
    """
    vRet= []
    i= 0
    iA= len(vArg)
    while (i < iA):
        try:
            dA= int(vArg[i])
            vRet.append(dA)
            i+= 1
        except ValueError:
            print ("Arg '%s' cannot be converted to int" % vArg[i])
            i= iA
    return vRet

#########################################################
def ReadArg_dict(dtArg, show= False, argv= None):
    """
    Purpose:
       Read the command line argument, filling in the dictionary elements by corresponding elements on the command line.
       So
            dtArg= {'a': 5, 'b': 'hello', 'c': True, 'd': [5, 6, 3], 'e': ['aa', 'bb', 'cc'], 'f': []}
            ir= ReadArg(dtArg)
       should read/check for all those keys. When a list is handed over, readarg should check for all elements until the next keyword? If an empty list is passed, same thing, though then by definition strings are read.

       Note that the routine is NOT finished probably, possibly the whole concept should be set up differently...

    Inputs:
        dtArg       dict, with label and value pairs
        argv        (optional, for debugging) list of arguments, instead of command line arguments

    Outputs:
        dtArg       dict, with values adapted as to the command line arguments

    Return value:
        ir          integer, number of keys found
    """
    lArg= sys.argv if (argv is None) else argv
    # dtArg= {'a': 5, 'b': 'hello', 'c': True, 'd': [5, 6, 3], 'e': ['aa', 'bb', 'cc'], 'f': [], 'args': 'exchange Xetra type depth date 2023-07-03 dir /mnt/etf_calc/tmp/2023-07 isin "IE000JBB8CR7 IE000QDFFK00 IE000SBHVL31 IE000Z8BHG02" label axa xyt whatever redo', 'isin': ''}
    # lArg= ['a', '45', 'hello', 'd', '345', '-2.4', '2.3', 'sdfsf', 'e', 'sdfsa', 'sdfadsfa', 'f', '345', '13', 'df']

    if ('args' in dtArg):
        # lArg= dtArg['args'].split(' ')        # Incorrect, as it splits quoted strings
        # lArg= [p for p in re.split("( |\\\".*?\\\"|'.*?')", dtArg['args']) if p.strip()]          # Correct, but a bit convoluted
        lArg= re.findall(r'[^"\s]\S*|".+?"', dtArg['args'])          # Also correct, simple findall?
        print ('Using arguments:', lArg)

    asK= list(dtArg.keys())
    iK= len(asK)
    iL= len(lArg)


    sK= None
    l0= l1= 0
    dtFound= {}
    while (l0 < iL):
        # print (f'Argument {l0}/{iL}...')
        if (lArg[l0] in asK):
            sK= lArg[l0]
            if isinstance(dtArg[sK], bool):
                l1= l0+1
                # print (f'{type(dtArg[sK])} {sK} at {l0}-{l1}')
            elif isinstance(dtArg[sK], (str, int, float)):
                l1= l0+1+1
                # print (f'{type(dtArg[sK])} {sK} at {l0}-{l1}')
            else:
                l1= l0+len(dtArg[sK])+1
                # print (f'{type(dtArg[sK])} {sK} at {l0}-{l1}')
            l1= min(l1, iL)
            dtFound[sK]= [l0, l1]
            l0= l1
        else:
            l0+= 1

    # print ('lArg: ', lArg)
    # print ('Locations found:', dtFound)

    # Read out the arguments
    # sK= 'a'
    ir= 0
    for sK in asK:
        bFound= (sK in dtFound)
        # print ('key %s, found= %i, arg=' % (sK, bFound), dtArg[sK])
        if (isinstance(dtArg[sK], bool)):
            dtArg[sK]= bFound
            ir+= 1
        elif ((isinstance(dtArg[sK], (list, np.ndarray))) and bFound):
            vLU= dtFound[sK]
            vOrg= dtArg[sK]
            vAns= lArg[vLU[0]+1:vLU[1]]
            bFloat= (len(vOrg) > 0) and isinstance(vOrg[0], float)
            bInt= (len(vOrg) > 0) and isinstance(vOrg[0], int)
            dtArg[sK]= ReadArg_float(vAns) if bFloat else ReadArg_int(vAns) if bInt else vAns
            ir+= 1
        elif (isinstance(dtArg[sK], str) and bFound):
            vLU= dtFound[sK]
            vAns= lArg[vLU[0]+1:vLU[1]]
            dtArg[sK]= vAns[0].strip('"') if (len(vAns)) else ''
            ir+= 1
        elif (isinstance(dtArg[sK], (float, int)) and bFound):
            vLU= dtFound[sK]
            vAns= lArg[vLU[0]+1:vLU[1]]
            dtArg[sK]= ReadArg_float(vAns[0:1])[0] if (len(vAns)) else None
            ir+= 1
        elif (bFound):
            print ('Key %s found, but I do not know what to do with it' % sK)
        # else:
        #     print ('Key %s not found, so not changed' % sK)

    if (show):
        print('ReadArg encountered arguments:')
        for sK in dtArg:
            print ('Key %s(reset=%i):' % (sK, sK in dtFound), dtArg[sK])

    return ir


###########################################################
### main()
def test():
    sArg= 'symbol fsd fsd thisfile'
    sArg= 'storeprices symbol fsdb fsd thisfile'
    dtArg= {'symbol': 'all',
            'fsd': 'otherfile',
            'storeprices': True}

    argv= sArg.split(' ')
    ir= ReadArg_dict(dtArg, show= True, argv= argv )

###########################################################
### main()
def main():
    # Magic numbers
    dA= 1000
    vB= [-1, -1, -1]
    sS= 'this is the default'
    tS= ['t1', 't2']

    dtArg= {'a': dA, 'b': vB, 's': sS, 't': tS}

    # dtArg['args']= 'a 5.0 b 1 2 3 s hallo t abc def'

    # Initialisation
    print ('Usage:\n  ipython readarg.py a 5.0 b 1 2 3 s hallo t abc def')
    print ('This would read a scalar a, vector b of (max) 3 elements, string s, and list of strings t.')
    print ('If arguments are not found, default values are filled in, if given')

    dA= ReadArg('a', 1)            # No default value
    vB= ReadArg('b', 3, vB)
    sS= ReadArg('s', -1, sS)
    tS= ReadArg('t', -2, tS)

    # Output
    print (f'A= {dA}', dA)
    print ('B=\n', vB)
    print (f's= {sS}', sS)
    print ('t= ', tS)

    ir= ReadArg(dtArg, show= True)
    print ('Using a dictionary: Found %i items, with values \n' % ir, dtArg)



###########################################################
### start main
if __name__ == "__main__":
    main()
