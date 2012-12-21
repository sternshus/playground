# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###Introduction:
# 
# In order to grade physicians’ performance on specific tasks a model of good and novice physicians must be created.  The specific steps to create this model are:  
# 1.  Vector Quantization: reducing the total number of dimensions and normalizing the data  
# 2.  Feature Segmentation: looking only at data while grasping objects  
# 3.  Push to HMM training   

# <markdowncell>

# #####Globals

# <markdowncell>

# Client & View:

# <codecell>

%load_ext autoreload
%autoreload 2
from datetime import datetime
import scipy.cluster.vq as vq
import vector_quantization_refactor as vqr
import segmentation_refactor as segr
import data_wrangling_page as dwp
import cStringIO
import json

# <codecell>

from IPython.parallel import Client
def getClient():
    ipclient = Client('/home/ubuntu/.starcluster/ipcluster/simcluster-us-east-1.json'
                      ,sshkey='/home/ubuntu/.ssh/simcluster.rsa'
                      ,packer='pickle')
    ipview = ipclient[:]
    return ipview, ipclient

ipview, ipclient = getClient()
print ipview
print ipclient

# <markdowncell>

# Raw Features:

# <codecell>

def getFeatures(table_name, task, dataset=None, sensors=None):
    sensors = sensors if sensors else getSensors(table_name)
    dataset = dataset if dataset else table_name
    thresholds = {task: vqr.getThresholds(table_name, dataset, task)}
    outliers = vqr.qOutliers(task, table_name, thresholds, sensors)
    results = dwp.queryGoogle(outliers)
    return results, thresholds

    #return (np.array([[vqr.floats(field['v']) for field in row['f']]for row in results['rows']]), thresholds[task])
dataset = 'timdataMatlab'
table_name = 'timdata'
task = 'suturing'
tasks = ['cutting', 'suturing', 'pegtransfer']
sensors = ['QgL', 'FgL','QgR', 'FgR'] 

import numpy as np
raw, thresholds = getFeatures(table_name, task, dataset, sensors)
test_features = np.array([[vqr.floats(field['v']) for field in row['f']]for row in raw['rows']])
print test_features

# <markdowncell>

# #####Vector Quantization

# <codecell>


# <codecell>

#features, thresholds = getFeatures('timdata', task, sensors)
def create_codebook(args):
    features, size = args
    
    import scipy.cluster as c
    
    left = c.vq.kmeans(features[0], size, 1) #if features[0] else None
    right = c.vq.kmeans(features[1], size, 1) #if features[1] else None
        
    return (left, right)

'''
#Normalize
print 'normalizing'
normFeatures = vqr.normalize(features, thresholds, sensors)
leftFeat = normFeatures[:,:2]
rightFeat = normFeatures[:,2:]

print size,
size = vqr.size_codebook(task)
print ' size: ', size

print 'starting timing'
start = datetime.now()
print start

run_view = ipview.map_async(create_codebook, [((leftFeat, rightFeat), size) for i in range(10)])
results = run_view.get()

end = datetime.now()
print end
print end - start 
'''
#resultsR = ipview.map_async(create_codebook, [(featuresR, size, i) for i in range(100)])
#lowest_distortion_left = min(results[0], key=itemgetter(1))
#lowest_distortion_right = min(results[1], key=itemgetter(1))
#left_name = 'left_tim_codebook_{0}_QgFg_thresh'.format(task)
#right_name = 'right_tim_codebook_{0}_QgFg_thresh'.format(task)

#json.dump(lowest_distortion_left[0].tolist(),  open(left_name, 'w'))
#json.dump(lowest_distortion_right[0].tolist(),  open(right_name, 'w'))

# <codecell>

def upload_codebook(left, right):
    data = cStringIO.StringIO()
    fields = dwp.getSchemaFields('codebooks')  
    
    data.write('{0},{1},"{2}"\n'.format(dataset + '_left', '19-SEPT-2012 1:33', json.dumps(left[0].tolist())))   
    data.write('{0},{1},"{2}"\n'.format(dataset + '_right', '19-SEPT-2012 1:33', json.dumps(right[0].tolist())))

    body = dwp.getBody(data.getvalue(), fields, 'codebooks', 'data'
                              , createDisposition='CREATE_IF_NEEDED'
                              , writeDisposition='WRITE_APPEND')
    dwp.loadTableFromCSV(body)
    


#upload_codebook(min(results[0], key=itemgetter(1)), min(results[1], key=itemgetter(1)))
#print tst.getvalue()
        
#dataset = 'timdataMatlab'
#table_name = 'timdata'

#tasks = ['cutting', 'suturing', 'pegtransfer']
#sensors = ['QgL', 'FgL','QgR', 'FgR'] 

#for task in task:

#features, thresholds = getFeatures(table_name, 'suturing', dataset, sensors)
#rv = upload_codebook('timdataMatlab', features, thresholds, task, sensors)

#print features[0]
#print thresholds
#print task

#size = vqr.size_codebook(task)
#print size
#print run_view.ready()


#x= run_view.get()

# <codecell>

'''from operator import itemgetter
import sys
print tst
s = tst.getvalue()
print sys.getsizeof(s)
print s'''
#distortions = [d[1] for d in results[0]]
#print len(distortions)
#mini= min(results[0], key=itemgetter(1))
#print mini[1]
#print distortions

# <markdowncell>

# #####Segmentation

# <codecell>

def segment(args):
    import numpy as np
    task, thresholds, kv = args
    task_threshold = thresholds.get(task, (0, 0))
    
    key = kv[0] 
    arr = np.array(kv[1])
    arr = arr.astype(float) #Convert from unicode to float
    
    lidx, = np.diff(arr[:, 0] > task_threshold[0]).nonzero()
    lidx = np.delete(lidx,-1) if len(lidx)%2==1 else lidx
    lidx = np.append(np.matrix(lidx[range(0,len(lidx),2)]).transpose(), np.matrix(lidx[range(1,len(lidx),2)]).transpose() , 1)
    lidx = np.squeeze(np.asarray(lidx)) #lidx = lidx.shape(-1,2) #puts idx into a two column matrix

    ridx, = np.diff(arr[:, 1] > task_threshold[0]).nonzero()
    ridx = np.delete(ridx,-1) if len(ridx)%2==1 else ridx
        
    ridx = np.append(np.matrix(ridx[range(0,len(ridx),2)]).transpose(), np.matrix(ridx[range(1,len(ridx),2)]).transpose() , 1)
    ridx = np.squeeze(np.asarray(ridx)) #ridx = ridx.shape(-1,2) #puts idx into a two column matrix

    #Check to make sure grasp is longer than time thresh:: IndexError: invalid index
    e = None
    try:
        left =  lidx[lidx[:, 1] - lidx[:, 0] > task_threshold[1]]
    except Exception as e:
        left = None
    
    try:
        right = ridx[ridx[:, 1] - ridx[:, 0] > task_threshold[1]]
    except Exception as e:
        right = None 
    
    return (key, left, right, e)

def run_segmentation(task, thresholds, data):
    start = datetime.now()
    print task, ': \n', start
    
    run_view = ipview.map_async(segment, [(task, thresholds, kv) for kv in data.iteritems()])
    results = run_view.get()
    
    end = datetime.now()
    print end
    print end - start
    return results
    
#put this in a function & loop
#function checks if segments exist
#if not it checks google
#if not there it kicks off the creation

# <codecell>

grasp_thresholds = {'suturing': (1.5, 0.1*30), 'cutting': (3, 0.5*30), 'pegtransfer': (3, 1*30)} 
#ipview, ipclient = getClient()

try:
    FgLR
except NameError:
    FgLR = None

if FgLR is None:
    print 'Loading data...'
    FgLR = segr.getData(table_name, task)
    print 'data got...'
    
#Segment FgL and R based on force and time thresholds
seg_results = run_segmentation(task, grasp_thresholds, FgLR)


# <codecell>

def split(arr, cond):
  return [arr[cond], arr[~cond]]


def splitFeatures(rest):
    features_split = {}
    
    while (rest[-1,0]!=rest[0,0]):   #.any()
        cut, rest = split (rest, rest[:,0]==rest[0,0])
        features_split[cut[0,0]] = cut[:,1:]
                     
    features_split[rest[0,0]] = rest[:,1:]
    return features_split 

testfiles = splitFeatures(test_features)


# <codecell>

#for item in seg_results:
    
from collections import defaultdict


def getGrasps(seg_results, testfiles):
    graspsegs = defaultdict(list) 
    
    for item in seg_results:
        #sensors = ['QgL', 'FgL','QgR', 'FgR'] 
        key = item[0]
        #print key
        xx = testfiles[item[0]]
        xx = xx.astype(float)
        graspsegs[key] = []
        try: #Get grasps for left side, checking there are left indices
            graspsegs[key] = []
            for i in range(len(item[1])):   
                #graspsegs[key]['left'][i] = xx[:,(0,2)][item[1][i,0] : item[1][i,1]] 
                graspsegs[key].append(['left', i, xx[:,(0,1)][item[1][i,0] : item[1][i,1]]] )
        except Exception as e:
            pass
            #graspsegs[key].append(['left', i, None ])
        
        try: #Get grasps for right side, checking there are left indices
            for i in range(len(item[2])):
                #graspsegs[key]['right'][i] = xx[:,(1,3)][item[2][i,0] : item[2][i,1]]
                graspsegs[key].append(['right', i, xx[:,(2,3)][item[1][i,0] : item[1][i,1]]] )
        except Exception as e:
            pass
            #graspsegs[key]['right'] = None
            #graspsegs[key].append(['right', i, None ])
        
    return graspsegs
        
graspsegs = getGrasps(seg_results, testfiles)

# <codecell>

#print graspsegs

# <codecell>

skills = open('/home/ubuntu/simulab/data/contentExpertLevel.csv')

skilldict = {'/home/ubuntu/simulab/data/CSVs/'+line.split(',')[0].strip(): line.split(',')[2].strip() for line in skills if line.strip()} 
#print skilldict

# <codecell>

def upload_getGrasps(graspsegs, skilldict):    
    data = cStringIO.StringIO()
    fields = dwp.getSchemaFields('segments')  

    for key, vlist in graspsegs.iteritems():
        skill = skilldict[key.strip()+'.txt']
        for v in vlist:
            if v:
                data.write( '{0},{1},{2},{3},{4},"{5}","{6}"\n'.format(key, 'timdata', skill, v[0], v[1], json.dumps(v[2][:,0].tolist()), json.dumps(v[2][:,1].tolist())) )
           # else:
                #print key, skill, v[0], v[1], json.dumps(v[2].tolist())
                
    #body = dwp.getBody(data.getvalue(), fields, 'segments', 'data'
    #                         , createDisposition='CREATE_IF_NEEDED'
    #                          , writeDisposition='WRITE_APPEND')
    #dwp.loadTableFromCSV(body)
    return data
data = upload_getGrasps(graspsegs, skilldict)

# <codecell>

def codebookApply(features, codebook):
    '''
    returns the code and distance for the codebook
    '''
    return vq.vq(features, codebook)

# <codecell>



'''
codebooks = dwp.queryGoogle('SELECT dataset,json FROM data.codebooks')

cbleft= codebooks['rows'][0]['f'][1]['v']
cbright= codebooks['rows'][1]['f'][1]['v']

cbl = np.array(json.loads(cbleft),dtype = np.float)
cbr = np.array(json.loads(cbright),dtype = np.float)
'''
def encodeSegSkillData(graspsegs,skilldict,cbl,cbr):
    encodedExpL = []#defaultdict(list) 
    encodedNovL = []#defaultdict(list) 
    encodedExpR = []#defaultdict(list) 
    encodedNovR = []#defaultdict(list) 
    for key, vlist in graspsegs.iteritems():
        skill = skilldict[key.strip()+'.txt']
        
        for v in vlist:
            if v:
                hand= v[0]
                if skill == 'Expert' and hand == 'left':
                    code,dist = vq.vq(v[2], cbl) 
                    encodedExpL.append(code) 
                elif skill == 'Novice' and hand == 'left':
                    code, dist= vq.vq(v[2], cbl) 
                    encodedNovL.append(code)
                elif skill == 'Expert' and hand == 'right':
                    code,dist= vq.vq(v[2], cbr) 
                    encodedExpR.append(code)
                elif skill == 'Novice' and hand == 'right':
                    code,dist = vq.vq(v[2], cbr) 
                    encodedNovR.append(code )
                        
            #end of v in vlist
        #end of key, vlist in graspsegs
        return encodedExpL, encodedNovL, encodedExpR, encodedNovL

encodedExpL, encodedNovL, encodedExpR, encodedNovR = encodeSegSkillData(graspsegs,skilldict,cbl,cbr)


            

# <codecell>

s = """"Your humble writer knows a little bit about a lot of things, but despite writing a fair amount about text processing (a book, for example), linguistic processing is a relatively novel area for me. Forgive me if I stumble through my explanations of the quite remarkable Natural Language Toolkit (NLTK), a wonderful tool for teaching, and working in, computational linguistics using Python. Computational linguistics, moreover, is closely related to the fields of artificial intelligence, language/speech recognition, translation, and grammar checking.\nWhat NLTK includes\nIt is natural to think of NLTK as a stacked series of layers that build on each other. Readers familiar with lexing and parsing of artificial languages (like, say, Python) will not have too much of a leap to understand the similar -- but deeper -- layers involved in natural language modeling.\nGlossary of terms\nCorpora: Collections of related texts. For example, the works of Shakespeare might, collectively, by called a corpus; the works of several authors, corpora.\nHistogram: The statistic distribution of the frequency of different words, letters, or other items within a data set.\nSyntagmatic: The study of syntagma; namely, the statistical relations in the contiguous occurrence of letters, words, or phrases in corpora.\nContext-free grammar: Type-2 in Noam Chomsky's hierarchy of the four types of formal grammars. See Resources for a thorough description.\nWhile NLTK comes with a number of corpora that have been pre-processed (often manually) to various degrees, conceptually each layer relies on the processing in the adjacent lower layer. Tokenization comes first; then words are tagged; then groups of words are parsed into grammatical elements, like noun phrases or sentences (according to one of several techniques, each with advantages and drawbacks); and finally sentences or other grammatical units can be classified. Along the way, NLTK gives you the ability to generate statistics about occurrences of various elements, and draw graphs that represent either the processing itself, or statistical aggregates in results.\nIn this article, you'll see some relatively fleshed-out examples from the lower-level capabilities, but most of the higher-level capabilities will be simply described abstractly. Let's now take the first steps past text processing, narrowly construed. """
sentences = s.split('.')[:-1]
seq = [map(lambda x:(x,''), ss.split(' ')) for ss in sentences]
symbols = list(set([ss[0] for sss in seq for ss in sss]))
states = range(5)
trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states,symbols=symbols)
m = trainer.train_unsupervised(seq)

# <codecell>

formattedNovR=[]
for seg in encodedNovR:
    segtuple = []
    for s in seg:
        segtuple.append((int(s),''))
    
    formattedNovR.append(segtuple)
    
import nltk 
nltkTrainer = nltk.tag.hmm.HiddenMarkovModelTrainer(range(15),range(cbr.shape[0]))
trained = nltkTrainer.train_unsupervised(formattedNovR, max_iterations=3)

# <codecell>

print trained

# <codecell>

a,b = formattedNovR[1][0]
print a
print type(int(a))

print type(range(cbr.shape[0])[0])

# <codecell>

M = len(range(cbr.shape[0])) 
symbol_dict = dict((range(cbr.shape[0])[i], i) for i in range(M)) 

for sequence in formattedNovR:
    sequence = list(sequence)
    T = len(sequence)
    for t in range(T):
        x = sequence[t][0]
    
        #a,b = formattedNovR[1][0]
        xi = symbol_dict[x]

# <codecell>

print seq[0][0]

# <codecell>


