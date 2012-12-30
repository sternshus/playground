# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# from 'A Tutorial on PCA' by Lindsay I Smith

# <codecell>

import math
import numpy
from numpy.fft import *
from pprint import pprint
import cmath

# <headingcell level=1>

# 1. Background

# <headingcell level=2>

# 1.1 Statistics

# <headingcell level=3>

# 1.1.1. Standard Deviation

# <codecell>

#sample
X = [1, 2, 4, 6, 12, 15, 25, 45, 68, 67, 65, 98]

# <codecell>

def mean(data):
    return float(sum(data)) / len(data)

# <codecell>

def sigma(sample):
    mu = mean(sample)
    return math.sqrt(sum([(x - mu)**2 for x in sample]) * 1./(len(sample) - 1.))

# <codecell>

#tests
set_1 = [0, 8, 12, 20]
print 'set 1:\n', set_1, '\nstandard deviation: ', sigma(set_1) 
set_2 = [8, 9, 11, 12]
print 'set 2:\n', set_2, '\nstandard deviation: ', sigma(set_2) 

# <headingcell level=3>

# 1.1.2. Variance

# <codecell>

def variance(sample):
    mu = mean(sample)
    return (sum([(x - mu)**2 for x in sample]) * 1.) / (len(sample) - 1.)

# <codecell>

#exercises:
a = [12, 23, 34, 44, 59, 70, 98]
b = [12, 15, 25, 27, 32, 88, 99]
c = [15, 35, 78, 82, 90, 95, 97]
for sample in [a, b, c]:
    print sample, '\nmean: ', mean(sample), '\nstandard deviation: ', sigma(sample), '\nvariance: ', variance(sample)

# <headingcell level=3>

# 1.1.3. Covariance

# <markdowncell>

# One of the bottlenecks is calculation of the covariance matrix.  A straightforward approach is n^2d^2. where d is the number of dimensions.
# A method for efficient calculation uses fft to calculate the covariance between 2 matrices resulting in nlog(n)d^2.

# <markdowncell>

# nlog(n) version:  
# fft code: http://jeremykun.com/2012/07/18/the-fast-fourier-transform/  
# fast covariance: http://research.google.com/pubs/pub36416.html  

# <codecell>

import numpy_pad as npp

epsilon = 1.e-14

def pad(n):
    p = 1
    while (p < n):
        p <<= 1
    p <<= 1
    npp.pad
    return  [0.] * (p - n)

def getmax(A):
    R = numpy.real(A)
    maxi = numpy.amax(R)
    mini = numpy.amin(R)
    maxcorr = if abs(maxi) > abs(mini) else mini
    index = np.argmax(R)
    #print 'maxcorr: ', maxcorr
    #pprint(A)
    
    return maxcorr, index
    
def fft_covariance(sample):
    '''
        returns the covariance matrix and index of each item for a dimension-list of sample-lists
        [d0, d1, ...,dd] where d0 is the first dimension containing n samples.
        n.b. numpy 1.7rc1 has numpy.pad to pad array.  currently running 1.6.2 (released version)
        so imported rc1 version.
    '''
        
    A = numpy.array(sample)
    d, n = A.shape
    Q = numpy.ones(1) if(d==2) else numpy.ones((d, d))
    idx = [[1]] if(d==2) else [list([1.] * d) for i in range(d)]
    nm1 = float(n - 1)
    #prep for fft
    A =  (A.transpose() - numpy.mean(A, axis=1)).transpose()
    A = npp.pad(A,(0,len(pad(n))), 'constant', constant_values=(0,0))[:d, : ]
    #2x2return
    if (d == 2):
        corr = ifft(fft(A[0, :]) * np.conjugate(fft(A[1, :])))
        maxcorr = getmax(corr)
        Q[0] =  maxcorr[0] / nm1
        idx[0] = maxcorr[1]
    else:
        for i in range(d):
            for j in range(i, d):
                corr = ifft(fft(A[i, :]) * np.conjugate(fft(A[j, : ])))
                maxcorr = getmax(corr)
                Q[i][j] = maxcorr[0] / nm1 
                idx[i][j] = maxcorr[1]
                if i != j:
                    Q[j][i] = Q[i][j]
    return Q, idx
    

# <codecell>

H = [9., 15., 25., 14., 10., 18., 0., 16., 5., 19., 16., 20.]
M = [39., 56., 93., 61., 50., 75., 32., 85., 42., 70., 66., 80.]
W = [120., 260., 490., 255., 150., 250., 0., 266., 88., 270., 270., 500. ]


p = fft_covariance([H, M])
print p

one = [1,2,1]
two = [-1, 1, 3]
three = [1, 3, -1]
c3 = covariance([one, two, three])
print 'C3'
pprint(c3)

c5 = fft_covariance([one, two, three])
print 'c5:'
pprint(c5)

# <codecell>

def omega(p, q, inverse=False):
    two = 2. if inverse else -2.
    return cmath.exp((two * cmath.pi * 1j * q) / p)

def my_fft(sample, inverse=False):
    n = len(sample)
    if n == 1:
        return sample
    else:
        Feven = fft([sample[i] for i in xrange(0, n, 2)])
        Fodd = fft([sample[i] for i in xrange(1, n, 2)])

        combined = [0] * n
        for m in xrange(n/2):
            combined[m] = Feven[m] + omega(n, -m, inverse) * Fodd[m]
            combined[m + n/2] = Feven[m] - omega(n, -m, inverse) * Fodd[m]

        return combined

# <codecell>


def pad(V):
    n = len(V)
    p = 1
    while (p < n):
        p <<= 1
    p <<= 1
    return  [0.] * (p - n)
    

# <markdowncell>

# d^2 version:
# To reduce the time the matrix is symmetrical complete for unordered pairs.

# <codecell>

def sansmu(V):
    vbar = mean(V)
    return [v - vbar for v in V]


def covariance(sample):
    '''
    calculate the covariance matrix for a given sample
    '''
    d = len(sample)
    n = len(sample[0])
    nm1 = float(n - 1)
    Q =  [list([1.] * d) for i in range(d)]
    mu0 = [sansmu(s) for s in sample]
    for i in range(d):
        for j in range(i, d):
            Q[i][j] = sum([x * y  for x,y in zip(mu0[i], mu0[j])]) / nm1
            if i != j:
                Q[j][i] = Q[i][j]
    return Q


# <codecell>

        
H = [9., 15., 25., 14., 10., 18., 0., 16., 5., 19., 16., 20.]
M = [39., 56., 93., 61., 50., 75., 32., 85., 42., 70., 66., 80.]

c1= covariance([H, M])
pprint(c1)

X = [10, 39, 19, 23, 28]
Y = [43, 13, 32, 21, 20]
c2 = covariance([X, Y])
pprint(c2)

one = [1,2,1]
two = [-1, 1, 3]
three = [1, 3, -1]
c3 = covariance([one, two, three])
pprint(c3)

c5 = fft_covariance([one, two, three])
print c5

sone = sigma(one)
stow = sigma(two)
sthree = sigma(three)

p = pad(one)
c41 = [c - mean(one) for c in one]
c42 = [c - mean(two) for c in one]
c43 = [c - mean(three) for c in one]
c41.extend(p)
c42.extend(p)
c43.extend(p)

c4a = np.array([c41, c42, c43])
print c4a.shape
print c4a.size

c4m = fftn(c4a, axes=[1])#, axes=1)
corrm =ifftn(c4m *(np.conjugate(c4m)), axes=[1])

realcorr = np.real(corrm)
pprint(realcorr)

pprint(realcorr / 2)

# <headingcell level=2>

# Matrix Algebra

# <codecell>


