# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Segmenting Positions by Speed 

# <markdowncell>

# Based on Timothy Kowalewski's Disseration

# <markdowncell>

# Problem:
# Calculate the speed of an object based on its position coordinates over time.  Segment the data based on speed threhsolds. 

# <codecell>

import numpy
import pandas
from pprint import pprint

# <markdowncell>

# Calculate the derivative (based on [holborodko](http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/)).  
#     
# Calculate the magnitude of each speed vector.
# 

# <codecell>

dprime = lambda (d): ((322. * (d[2:]-d[0 :-2])[4:-4]) + (256. * (d[4:]-d[0:-4])[3:-3]) + (39. * (d[6:]-d[0:-6])[2:-2]) - (32. * (d[8:]-d[0:-8])[1:-1]) - (11. * (d[10:]-d[0:-10]))) / 1536./.03
 
magnitude = lambda(positions): numpy.sum(dprime(positions)**2, axis=-1)**0.5
    

# <markdowncell>

# return indicies where the difference between two contiguous rows is over the threshold:

# <codecell>

def deltas(magnitudes, limena):
    threshold, hysterisis = limena
    x = numpy.diff(magnitudes > threshold)
    idx, = x.nonzero()
    #make sure even so start/finish for each entry
    #drop last if odd (between not closed --thus invalid)
    if len(idx) % 2 != 0:
        idx = numpy.delete(idx, -1)
    idx.shape = (-1, 2)
    return idx[idx[:, 1] - idx[:, 0] > hysterisis]

# <codecell>

def segmentation(coordinates, limena):
    start, end = (0, 1)
    idx = deltas(magnitude(coordinates), limena)
    return [coordinates[index[start] : index[end]] for index in idx]
    
segments = segmentation(left, (1.5, 10))
 

# <markdowncell>

# Load data:
# 
# Data is sorted by time (in original query)
# Generate left/right tool coordinates.

# <codecell>

data_type = {'names' : ['Time', 'xL', 'yL', 'zL', 'xR', 'yR', 'zR']
        ,'formats' : [numpy.float, numpy.float, numpy.float, numpy.float, numpy.float, numpy.float, numpy.float]}

data = numpy.loadtxt('data/positions.csv', dtype=data_type, skiprows=1, delimiter=',')
left = numpy.array(data[['xL','yL', 'zL']]).view(numpy.float).reshape(-1, 3)
right = numpy.array(data[['xR','yR', 'zR']]).view(numpy.float).reshape(-1, 3)


# <codecell>

for idx in left_deltas:
    start = 0
    end = 1
    print idx[start], 'to', idx[end], '\n', left[idx[start] : idx[end]]
    #print left[idx[start]], ' to ', left[idx[end]]

# <markdowncell>

# I was told that my deltas algorithm returned incorrect values.  Here is a test to verify the differences 

# <codecell>

#test using matrix squeezing:
def matrix_deltas(magnitudes, limena):
    threshold, hysterisis = limena
    x = numpy.diff(magnitudes > threshold)
    idx, = x.nonzero()
    #make sure even so start/finish for each entry
    #drop last if odd (between not closed)
    if len(idx) % 2 != 0:
        idx = numpy.delete(idx, -1)
    idx = numpy.append(numpy.matrix(idx[range(0,len(idx),2)]).transpose(), numpy.matrix(idx[range(1,len(idx),2)]).transpose() , 1)
    return numpy.squeeze(numpy.asarray(idx))
    

# <codecell>

#test:
mleft = magnitude(left)
mright = magnitude(right)

left_deltas = deltas(mleft, (1.5, 10))
left_matrix = matrix_deltas(mleft, (1.5, 10))
right_deltas = deltas(mright, (1.5, 10))
right_matrix = matrix_deltas(mright, (1.5, 10))
lcomp = thresh_mia - thresh_
max_diff = numpy.max(numpy.abs(comparison))
if max_diff > 1e-6:
    print 'proof the diff'
else:
    print 'the arrays contain same values'
print len(left_deltas)

# <markdowncell>

# There is no difference between the two numpy arrays.  Because I don't coerce the numpy array into the matrix form, sticking with deltas.

