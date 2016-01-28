# Utility script to compute a simple confusion matrix.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
import numpy as np

class Confmat(object):
    def __init__(self, expected, predicted):
        classes = list(np.sort(np.unique(np.array([expected, predicted]))))
        conf = np.zeros( (len(classes), len(classes)) )
        for e,p in zip(expected, predicted):
            conf[classes.index(e), classes.index(p)] += 1
        
        self.classes = classes
        self.confmat = conf
    
    def correct(self):
        return np.sum(np.diag(self.confmat))

    def incorrect(self):
        return np.sum(self.confmat) - self.correct()
    
    def accuracy(self):
        return self.correct() / np.sum(self.confmat)
    
    def __str__(self):
        classes, conf = self.classes, self.confmat
        s  = "       | Predicted\n"
        s += "Expect |" + "".join([ "%10d" % i for i in xrange(0, len(classes)) ]) + "\n"
        s += "-" * (8 + 10 * len(classes)) + "\n"
        for e in xrange(0, len(classes)):
            s += ("%6d |" % e) + ("".join([ "%10d" % conf[e][p] for p in xrange(0, len(classes)) ])) + "\n"
        return s

