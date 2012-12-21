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


# <markdowncell>

# Calculate the derivative (based on [holborodko](http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/)
# Calculate the magnitude of each speed vector
# 

# <codecell>

dprime = lambda (d): ((322. * (d[2:]-d[0 :-2])[4:-4]) + (256. * (d[4:]-d[0:-4])[3:-3]) + (39. * (d[6:]-d[0:-6])[2:-2]) - (32. * (d[8:]-d[0:-8])[1:-1]) - (11. * (d[10:]-d[0:-10]))) / 1536./.03
 
magnitude = lambda(positions): numpy.sum(dprime(positions)**2, axis=-1)**0.5
    

# <markdowncell>

# magnitude

# <codecell>


# <codecell>


# <codecell>


# <codecell>


