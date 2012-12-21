# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%load_ext autoreload
#%autoreload 2

from scipy.cluster import vq
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

# <rawcell>

# Initial Results (9 files)
# cutting:
# 2012-09-04 07:59:33.246089 -starttime
# 1.41955496309 lowest distortion
# 2012-09-04 08:07:55.862776 -endtime
# 0:08:22.616687 duration.
# 
# suturing:
# 2012-09-04 08:20:37.537422
# 1.0831198433
# 2012-09-04 08:27:00.092817
# 0:06:22.555395
# 
# 
# pegtransfer:
# 
# 2012-09-04 08:48:08.336336
# 1.81233551748
# 2012-09-04 08:55:38.810838
# 0:07:30.474502

# <rawcell>

# Results (558 files)
# Cluster: 19 machines 1 master
# Peg Transfer: task
# 2012-09-05 04:52:47.095077 start time
# 1.62632084185 lowest distortion
# 2012-09-05 04:56:50.470700 endtime
# 0:04:03.375623 duration
# 
# Cutting:
# 2012-09-05 05:23:56.818599
# 1.5186917313
# 2012-09-05 05:28:40.822344
# 0:04:44.003745
# 
# Suturing:
# 2012-09-05 05:31:51.683836
# 1.42123512532
# 2012-09-05 05:35:45.736178
# 0:03:54.052342

# <rawcell>

# 550+ samples features = QgFg (L/R)
# 
# suturing: 
# 2012-09-06 11:33:43.069555
# distortion_left 0.0782727991679
# distortion_right 0.0777836563702
# 2012-09-06 12:06:13.599771
# 0:32:30.530216  
# 
# cutting:  
# 2012-09-06 11:00:54.851233
# distortion_left 0.0679669846974
# distortion_right 0.0666527180284
# 2012-09-06 11:31:46.766268
# 0:30:51.915035
# 
# peg transfer:
# 2012-09-06 10:33:05.906191
# distortion_left 0.0995422580172
# distortion_right 0.0994004187369
# 2012-09-06 10:58:57.604016
# 0:25:51.697825

# <rawcell>

# Suturing Qg and Fg:
# 2012-09-10 20:20:03.047888
# distortion_left 0.0426903171195
# distortion_right 0.0405832744826
# 2012-09-10 20:40:35.902251
# 0:20:32.854363
# 
# Cutting:
# 2012-09-11 20:35:59.647566
# distortion_left 0.0419672938211
# distortion_right 0.0403074226113
# 2012-09-11 20:56:47.522063
# 0:20:47.874497

# <rawcell>

# 1000 kmeans runs X 1 run per time
# using matlab quantiles & python normalization
# This is not getting lower.
# 2012-09-12 04:08:33.913603
# distortion_left 0.0401090267562
# distortion_right 0.0412857620919
# 2012-09-12 07:12:44.583323
# 3:04:10.669720

# <markdowncell>

# #Create Codebook

# <codecell>

print 'starting client'
from IPython.parallel import Client
ipclient = Client('/home/ubuntu/.starcluster/ipcluster/simcluster-us-east-1.json'
            ,sshkey='/home/ubuntu/.ssh/simcluster.rsa'
            ,packer='pickle')
ipview = ipclient[:]

print 'getting features'
task = 'suturing'
size = size_codebook(task)
sensors = ['QgL', 'FgL','QgR', 'FgR'] 

features, thresholds = getFeatures('timdata', task, sensors)

#Normalize
normFeatures = normalizeTim(features, thresholds, sensors)
leftFeat = normFeatures[:,:2]
rightFeat = normFeatures[:,2:]

def create_codebook(args):
    features, size, i = args
    
    import scipy.cluster as c
    left = c.vq.kmeans((features[0], size)
    right = c.vq.kmeans(features[1], size)
    return (left, right)

print 'starting timing'
start = datetime.now()
print start

run_view = ipview.map_async(create_codebook, [((leftFeat, rightFeat), size, i, 1) for i in range(100)])
results = run_view.get()
#resultsR = ipview.map_async(create_codebook, [(featuresR, size, i) for i in range(100)])
lowest_distortion_left = min(results[0], key=itemgetter(1))
lowest_distortion_right = min(results[1], key=itemgetter(1))
json.dump(lowest_distortion_left[0].tolist(),  open('left_tim_codebook_{0}_QgFg_thresh'.format(task), 'w'))
json.dump(lowest_distortion_right[0].tolist(),  open('right_tim_codebook_{0}_QgFg_thresh'.format(task), 'w'))

print 'distortion_left', lowest_distortion_left[1]
print 'distortion_right', lowest_distortion_right[1]
end = datetime.now()
print end
print end - start

# <codecell>

#print leftFeat[:10,:]
#print '\n', features[:10,:2]

#import matplotlib.pyplot as plt
#plt.hist(leftFeat,50)
#print results
#print normFeatures
#np.savetxt("normFeatures.csv", normFeatures, delimiter=",")

# <codecell>

#Normalize all features by mapping 2nd and 98th percentiles to -1 and 1 
#(features must have outliers removed)
def normalize(features,thresholds,sensors):
    (rows,cols) = features.shape
    normFeatures = np.zeros((rows,cols))

    #print rows, cols
    for col in range(cols):
        feature = features[:,col]
        print 'normalizing', sensors[col]+'...'
    
        #Retrieve 2nd and 98th thresholds
        lowNormT =  float(thresholds[sensors[col]]['lowNorm'])
        highNormT = float(thresholds[sensors[col]]['highNorm'])
    
        #Perform normalization as based on Tim Code (i.e. Map 2nd and 98th perc. to [-1 1]
        normFeatures[:,col] = (feature - lowNormT)* (2/(highNormT-lowNormT)) - 1
    
    #normFeatures now normalized version of features
    print 'Done'   

    return normFeatures
    

# <codecell>

quantiles = namedtuple('quantiles', 'lowOutlier lowNorm highNorm highOutlier')
permil = quantiles(4, 19, 979, 994)

# <codecell>

def size_codebook(task):
    if task in ['cutting']: return 67
    if task in ['suturing']: return 70
    if task in ['pegtransfer']: return 57
    return 70

# <codecell>

def codebookApply(features, codebook):
    '''
    returns the code and distance for the codebook
    '''
    return vq.vq(features, codebook)

# <codecell>

def floats(s):
    try:
        return float(s)
    except Exception as e:
        return s
        #print 'error :', e
        #return np.NaN

# <codecell>

def getSensors(table_name):
    fields = json.loads(getSchemaFields(table_name))
    sensors =  [field['name'] for field in fields][4:]
    return sensors

#print getSensors('timdata')

# <codecell>

def getTasks(table_name):
    tasks = queryGoogle("SELECT task from data.{0} GROUP BY task".format(table_name))
    return [task['f'][0]['v'] for task in tasks['rows']]

#print getTasks('timdata')

# <markdowncell>

#  Gr -- includes grasp angle Qg and grasp force Fg (This requires no estimated derivatives or tip position, but provides very little SIG)

# <markdowncell>

# 2. SpdAcc-dRta -- Scalar speed, acceleration, and rate of rotation (shown to generate the most SIG)

# <codecell>

def getFeatures(table_name, task, dataset=None, sensors=None):
    sensors = sensors if sensors else getSensors(table_name)
    dataset = dataset if dataset else table_name
    thresholds = {task: getThresholds(table_name, dataset, task)}
    outliers = qOutliers(task, table_name, thresholds, sensors)
    results = queryGoogle(outliers)

    return (np.array([[floats(field['v']) for field in row['f']]for row in results['rows']]), thresholds[task])

# <codecell>

def getThresholds(table_name, dataset=None, task_type=None):
    dataset = dataset if dataset else table_name
    thresholds = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    #data = queryTableData('data', 'thresholds')
    qs = "SELECT task, table_name, threshold, sensor_name, sensor_value FROM data.thresholds WHERE table_name='{0}'".format(dataset)
    data = queryGoogle(qs)
    for row in data['rows']:
        cells = row['f']
        task = 'pegtransfer' if cells[0]['v']=='PegTx' else cells[0]['v'].lower()
        ttype = cells[2]['v']
        sensor = cells[3]['v']
        thresholds[task][sensor][ttype] = cells[4]['v']
    
    if task_type:
        return thresholds.get(task_type, 'ERROR: task not found')
    
    return thresholds


#print getThresholds('timdata')

# <codecell>

def createThresholds(table_name):
    sensors = getSensors(table_name)
    thresholds = cStringIO.StringIO()
    for task in getTasks(table_name):
        quantiles = {sensor: queryGoogle(qQuantiles(sensor, table_name, task)) for sensor in sensors}
        for field in permil._fields:
            p = getattr(permil, field)
            for sensor in sensors:
                thresholds.write('{0},{1},{2},{3},{4}\n'.format(task, table_name, field, sensor, quantiles[sensor]['rows'][p]['f'][0]['v']))
        #thresholds = cStringIO.StringIO(open('thresholds.csv').read()) #to hardcode thresholds from matlab
    return thresholds

#print createThresholds('timdataMatlab').getvalue()

# <codecell>

def createThresholdsTable(table_name):
    data = createThresholds(table_name)
    fields = getSchemaFields('thresholds')       
    body = getBody(data.getvalue(), fields, 'thresholds', 'data'
                               , createDisposition='CREATE_IF_NEEDED'
                               , writeDisposition='WRITE_APPEND')
    loadTableFromCSV(body)
#createThresholdsTable('timdata')

# <markdowncell>

# #Utilities

# <markdowncell>

# ##query to get Quantiles

# <codecell>

    
def qQuantiles(sensor, table_name, task): 
    return "SELECT QUANTILES({0}, 1000) as q{0} FROM [data.{1}] WHERE task=='{2}'".format(sensor, table_name, task)

left = ['QgL', 'FgL']
right = ['QgR', 'FgR'] 
sensors = ['QgL', 'FgL', 'QgR', 'FgR']
d =  {sensor: queryGoogle(qQuantiles(sensor, 'timdata', 'suturing')) for sensor in sensors}
print d

# <codecell>

def qThresholdSchema(task, table_name):
    sensors = getSensors(table_name)
    fields = ', '.join(sensors)
    return "SELECT threshold, task, {0} FROM data.schemata WHERE task=='{1}' AND table_name=='{2}'".format(fields, task, table_name)

# <markdowncell>

# ##query to remove outliers based on thresholds

# <codecell>

def qOutliers(task, table_name, thresholds, sensors=None):
    sensors = sensors if sensors else getSensors(table_name) 
    SELECT = ("SELECT key, ")   
    select = [] 
    FROM = (" FROM [data."+ table_name +"] ")
    WHERE = ("WHERE task='{0}' AND ".format(task))
    where = []
    for sensor in sensors:
        select.append(sensor)
        if abs(float(thresholds[task][sensor]['lowNorm']) - float(thresholds[task][sensor]['highNorm'])) > 0.01 or \
           abs(float(thresholds[task][sensor]['lowOutlier']) - float(thresholds[task][sensor]['highOutlier'])) > 0.01:
            where.append("({0} > {1} AND {0} < {2})\n".format(sensor, 
                                                             thresholds[task][sensor]['lowOutlier'], 
                                                             thresholds[task][sensor]['highOutlier']))
    
    return SELECT + (', '.join(select)) + FROM + WHERE + ' AND '.join(where) 


#sensors = ['QgL', 'FgL', 'QgR', 'FgR']
#print qOutliers('suturing', 'timdata', getThresholds('timdata', 'timdataMatlab'), sensors)

# <codecell>

f='''sensors = ['QgL', 'FgL', 'QgR', 'FgR']
#print thresholds[sensors[0]]['lowNorm'], thresholds[sensors[0]]['highNorm']
print thresholds[sensors[0]]['lowOutlier'], thresholds[sensors[0]]['lowNorm'], thresholds[sensors[0]]['highNorm'], thresholds[sensors[0]]['highOutlier']

#print thresholds[sensors[1]]['lowNorm'], thresholds[sensors[1]]['highNorm']
#print thresholds[sensors[1]]['lowOutlier'], thresholds[sensors[1]]['highOutlier']'''

