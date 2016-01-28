#!/usr/bin/env python
import sys
import numpy
import scipy.io as sio
import gzip, cPickle

m = sio.loadmat('filters16.mat')

print m.keys()

print m['f_lm'].shape
print m['f_s'].shape

m['f_lm'] = numpy.rollaxis(m['f_lm'], 2).reshape([48,1,16,16])
m['f_s'] = numpy.rollaxis(m['f_s'], 2).reshape([13,1,15,15])

print m['f_lm'].shape
print m['f_s'].shape

# with open('../20140221-ild-random-filters/orig-10x10x4.pkl', 'r') as f:
#   filters = cPickle.load(f)

#   for k in sorted(filters.keys()):
#     print k, filters[k].shape

f_lm = { 'W': m['f_lm'], 'bh': numpy.zeros([48]) }
f_s = { 'W': m['f_s'], 'bh': numpy.zeros([13]) }

with open('filter-bank-LM-16x16x48.pkl', 'w') as f:
  cPickle.dump(f_lm, f)
with open('filter-bank-S-15x15x13.pkl', 'w') as f:
  cPickle.dump(f_s, f)

