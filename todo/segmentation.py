# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import json
from collections import defaultdict, namedtuple
from operator import itemgetter
import cStringIO

import data_wrangling_page as dw
from data_wrangling_page import getBody
from data_wrangling_page import httpGoogle
from data_wrangling_page import getSchemaFields

from data_wrangling_page import loadTableFromCSV
from data_wrangling_page import queryTableData
from data_wrangling_page import queryGoogle

from datetime import datetime

# <headingcell level=2>

# Segmentation Script

# <codecell>

#GrspVarTh{PegTx}    = [ 3 3   3 3    1];   %{Qo Qc Fo Fc Tth];
#GrspVarTh{Suturing} = [ 4 3   3 3    .1];  %{Qo Qc Fo Fc Tth]; 
#GrspVarTh{Cutting}  = [ 3 3  1.5 1.5 .5];  %{Qo Qc Fo Fc Tth]; 


def deltas(column, task_threshold):
    deltas = np.diff(column > task_threshold[0]) 
    idx, = deltas.nonzero() #+ 1 
    idx.shape = (-1, 2) #puts idx into a two column matrix
    return idx[idx[:, 1] - idx[:, 0] > task_threshold[1]] 

def get_idx(arr, col):
    idx, = np.diff(arr[:, col] > task_threshold[0]).nonzero()
    if len(idx)%2==1:
        idx = np.delete(idx,-1)
    #lidx = lidx.shape(-1,2) #puts idx into a two column matrix (left = start, right = end)
    idx = np.append(np.matrix(idx[range(0,len(idx),2)]).transpose(), np.matrix(idx[range(1,len(idx),2)]).transpose() , 1)
    return np.squeeze(np.asarray(idx))
    
    

#Perform segmentation for specific task, return binary index of open/closed
def segment(args):
    import numpy as np #note for parallelization, this has to be here --TODO: framework script to set environment variables and global imports
    task, thresholds, kv = args
    
    task_threshold = thresholds.get(task, (0, 0))

    
    key = kv[0] #(-1, 2) #puts idx into a two column matrix
    arr = np.array(kv[1])
    arr = arr.astype(float) #Convert from unicode to float
    
    lidx = get_idx(arr, 0)
    ridx = get_idx(arr, 1)
   
    #Check to make sure grasp is longer than time thresh:: IndexError: invalid index
    try:
        left =  lidx[lidx[:, 1] - lidx[:, 0] > task_threshold[1]]
        right = ridx[ridx[:, 1] - ridx[:, 0] > task_threshold[1]]
    except Exception as e:
        return (key, e, lidx, ridx, )
    
    return (key, left, right)


# <headingcell level=2>

# Functions

# <codecell>

def getData(table_name, task):
    query_string = "SELECT key, FgL, FgR FROM [data.{0}] WHERE task = '{1}'".format(table_name, task)
    rows = queryGoogle(query_string)
    data = defaultdict(list)
    
    for row in rows['rows']:
        key = row['f'][0]['v']
        fgl = row['f'][1]['v']
        fgr = row['f'][2]['v']
        data[key].append([fgl, fgr])
    return data

#x = getData('timdata','cutting')

# <codecell>

def getQgLR(table_name, task):
    query_string = "SELECT key, QgL, FgL, QgR, FgR FROM [data.{0}] WHERE task = '{1}'".format(table_name, task)
    rows = queryGoogle(query_string)
    data = defaultdict(list)
    
    for row in rows['rows']:
        key = row['f'][0]['v']
        qgl = row['f'][1]['v']
        fgl = row['f'][2]['v']
        qgr = row['f'][3]['v']
        fgr = row['f'][4]['v']
        data[key].append([qgl, qgr])
    return data

#x = getData('timdata','cutting')

# <codecell>

def getIndicies(items, task): 
    arr = np.array(items)
    left =  deltaIndicies(arr[:, 0],task)
    right = deltaIndicies(arr[:, 1],task)
    return (left, right)
#x = [[1,2,3], [4,5,6]]  
#idxs = getIndicies(x, 'cutting')

# <codecell>

def deltaIndicies(column, task):
        #Provide thresholds for each task
    if (task == 'suturing'):
        Fth = 1.5 #Threshold for grasp
        timeth = 0.1*30 #Time threshold - 30Hz is sampling rate of EDGE
    elif (task == 'cutting'):
        Fth = 3 
        timeth = 0.5*30
    elif (task == 'pegtransfer'):
        Fth = 3
        timeth = 1*30
        
    deltas = np.diff(column > Fth) 
    idx, = deltas.nonzero() #+ 1 
    idx.shape = (-1, 2) #puts idx into a two column matrix
    
    return idx[idx[:, 1] - idx[:, 0] > timeth] 
#x = np.array([[1,2,3,0,0,0,0,3,1,1,1,0,0,0]]) 
#deltaIndicies(x, 'pegtransfer')

