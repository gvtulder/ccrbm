# Loads a trained classification RBM and computes the classification results.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
from collections import OrderedDict
import scipy.io as sio
import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters, prediction, objectives, activation_functions
import theano
import theano.tensor as T
import numpy as np
import gzip, cPickle, time
import json, sys, os, time, os.path
import gc

from morb import activation_functions
from theano.tensor.nnet import conv

import sklearn.datasets
import sklearn.ensemble
import sklearn.metrics
import sklearn.preprocessing

import borderconvparameters

from theano import ProfileMode
mode = None

theano.config.floatX = 'float32'

# do not use scientific notation
np.set_printoptions(suppress=True)

############################################
############################################

mb_size = 1 # 10 # 1


import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("--previous-layer", metavar="PKL", type=str, nargs="+")
parser.add_argument("--subsample", metavar="PX", type=int,
                    help="subsample first layer",
                    default=1)
parser.add_argument("--train-scans", metavar="S", type=str,
                    help="scans used for training",
                    default="069,048")
parser.add_argument("--test-scans", metavar="S", type=str,
                    help="scans used for testing",
                    default="002,007")
parser.add_argument("--n-states", metavar="N", type=int,
                    help="number of classes",
                    default=4)
parser.add_argument("--skip-normalisation", action="store_true")
parser.add_argument("--global-normalisation", action="store_true",
                    help="Use global normalisation, not per-patch")
parser.add_argument("--save-predictions", metavar="F", type=str,
                    help="write labels to file",
                    default=None)
parser.add_argument("--subset", metavar="PROP", type=float,
                    help="Train and test on a small subset",
                    default=None)
parser.add_argument("--rng-seed", metavar="SEED", type=int, default=123)
parser.add_argument("--convolution-type", required=True,
                    choices=["no", "full", "fullnoborder"])
args = parser.parse_args()



print "previous layer: ", args.previous_layer
print "skip normalis.: ", args.skip_normalisation
print "global normal.: ", args.global_normalisation
print "subset:         ", args.subset
print "rng seed:       ", args.rng_seed
print "convolution:    ", args.convolution_type


numpy_rng = np.random.RandomState(args.rng_seed)



############################################
# DATA
############################################

# load data
print ">> Loading dataset..."
train_scans = args.train_scans.split(",")
test_scans = args.test_scans.split(",")

train_data = []
train_labels = []
train_src = []
for s in train_scans:
  if ".mat" in s:
    m = sio.loadmat(s)
  else:
    m = sio.loadmat("SALD-cells-with-borders/"+s+".mat")
  train_data.append(np.transpose(np.double(m['neighbourhoods']).astype(theano.config.floatX)))
  train_labels.append(np.transpose(m['labels'].astype(theano.config.floatX)))
  pic_w = int(np.sqrt(m['cells'].shape[0]))
  train_src += ([ s ] * m['labels'].shape[1])
  m = None
train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)
train_src = np.array(train_src)

# remove any extra classes
subset = (train_labels[:,0] <= args.n_states)
train_data = train_data[subset]
train_labels = train_labels[subset]
train_src = train_src[subset]

order = numpy_rng.permutation(train_data.shape[0])
n = train_data.shape[0]
if not args.subset is None:
  n = min(n, int(args.subset * n))
order = order[0:(np.floor_divide(n, mb_size) * mb_size)]
train_data = train_data[order]
train_labels = train_labels[order]
train_src = train_src[order]

print "train shape:", train_data.shape
print "train shape:", train_labels.shape
print "train shape:", train_src.shape

# load test scans
test_data = []
test_labels = []
test_src = []
for s in test_scans:
  if ".mat" in s:
    m = sio.loadmat(s)
  else:
    m = sio.loadmat("SALD-cells-with-borders/"+s+".mat")
  test_data.append(np.transpose(np.double(m['neighbourhoods']).astype(theano.config.floatX)))
  test_labels.append(np.transpose(m['labels'].astype(theano.config.floatX)))
  test_src += ([ s ] * m['labels'].shape[1])
  m = None
test_data = np.concatenate(test_data)
test_labels = np.concatenate(test_labels)
test_src = np.array(test_src)

# remove any extra classes
subset = (test_labels[:,0] <= args.n_states)
test_data = test_data[subset]
test_labels = test_labels[subset]
test_src = test_src[subset]

order = numpy_rng.permutation(test_data.shape[0])
n = test_data.shape[0]
order = order[0:(np.floor_divide(n, mb_size) * mb_size)]
test_data = test_data[order]
test_labels = test_labels[order]
test_src = test_src[order]

print "test shape: ", test_data.shape
print "test shape: ", test_labels.shape
print "test shape: ", test_src.shape
print

train_distr = [ sum(sum(train_labels==i)) for i in np.sort(np.unique(train_labels)) ]
test_distr  = [ sum(sum(test_labels==i)) for i in np.sort(np.unique(test_labels)) ]
train_priors = np.asarray(train_distr, dtype=float) / sum(train_distr)
test_priors = np.asarray(test_distr, dtype=float) / sum(test_distr)
print "train distribution: ", train_distr
print "train priors:       ", train_priors
print "test distribution:  ", test_distr
print "test priors:        ", test_priors
print
print "largest-class classifier accuracy: train ", max(train_priors)
print "largest-class classifier accuracy: test  ", max(test_priors)
print "random classifier accuracy train:        ", sum(train_priors ** 2)
print "random classifier accuracy test:         ", sum(test_priors ** 2)
print

# garbage collection
gc.collect()


############################################
# CONVERT TO INPUTS
############################################

############################################
# PREPARE FOR CONVOLUTION
############################################

pic_w_from_data = int(np.sqrt(train_data.shape[1]))
if train_data.ndim == 2:
  train_set_x = train_data.reshape((train_data.shape[0], 1, pic_w_from_data, pic_w_from_data))
else:
  train_set_x = train_data.reshape([train_data.shape[0], 1] + list(train_data.shape[1:100]))
if test_data.ndim == 2:
  test_set_x = test_data.reshape((test_data.shape[0], 1, pic_w_from_data, pic_w_from_data))
else:
  test_set_x = test_data.reshape([test_data.shape[0], 1] + list(test_data.shape[1:100]))

# release
train_data = None
test_data = None


############################################
# NORMALISE (if required)
############################################

if args.global_normalisation:
  # normalise / whiten
  global_mu = np.mean(train_set_x)
  global_sigma = np.std(train_set_x)
  train_set_x -= global_mu
  train_set_x /= (0.25 * global_sigma)
  test_set_x -= global_mu
  test_set_x /= (0.25 * global_sigma)

elif not args.skip_normalisation:
  # normalise / whiten
  print ">> Normalising training data..."
  n_samples = train_set_x.shape[0]
  train_set_rows = train_set_x.reshape(n_samples, train_set_x.shape[1], -1)
  mu = np.mean(train_set_rows, axis=2).reshape(n_samples, train_set_x.shape[1], 1, 1)
  sigma = np.std(train_set_rows, axis=2).reshape(n_samples, train_set_x.shape[1], 1, 1)
  train_set_x -= mu
  train_set_x /= (0.25 * sigma)
  # release
  train_set_rows = None

  print ">> Normalising testing data..."
  n_samples = test_set_x.shape[0]
  test_set_rows = test_set_x.reshape(n_samples, test_set_x.shape[1], -1)
  mu = np.mean(test_set_rows, axis=2).reshape(n_samples, test_set_x.shape[1], 1, 1)
  sigma = np.std(test_set_rows, axis=2).reshape(n_samples, test_set_x.shape[1], 1, 1)
  test_set_x -= mu
  test_set_x /= (0.25 * sigma)
  # release
  test_set_rows = None

# garbage collection
gc.collect()

def memory_efficient_std(data):
  # computes np.std(data, axis=1)
  std = np.zeros(data.shape[0], data.dtype)
  for i in xrange(data.shape[0]):
    std[i] = np.std(data[i])
  return std

def predict_labels(args, data_set_x, normalisation={}):
  ############################################
  # APPLY FIRST LAYER CONVO
  ############################################

  assert len(args.previous_layer) == 1

  for prev_layer in args.previous_layer:
    print ">> Processing layer: ", prev_layer

    data_set_x_conv_collect = []

    with open(prev_layer, "r") as f:
      prev_layer_params = cPickle.load(f)

      prev_W = prev_layer_params["W"]
      prev_U = prev_layer_params["U"]
      prev_bv = prev_layer_params["bv"]
      prev_bh = prev_layer_params["bh"]
      prev_by = prev_layer_params["by"]

      print "       prev_W.shape:  ", prev_W.shape
      print "       prev_U.shape:  ", prev_U.shape
      print "       prev_bv.shape: ", prev_bv.shape
      print "       prev_bh.shape: ", prev_bh.shape
      print "       prev_by.shape: ", prev_by.shape

      filter_height = prev_W.shape[2]
      filter_width = prev_W.shape[3]
      pic_h = pic_w

      if args.convolution_type == "fullnoborder":
        print "   Removing neighbourhoods:"
        print "     before:            ", data_set_x.shape

        # cut borders
        margin = (data_set_x.shape[2] - pic_w) / 2
        data_set_x = data_set_x[:,:,(margin):(pic_w + margin),(margin):(pic_w + margin)]

        print "     after:             ", data_set_x.shape

      elif args.convolution_type == "full":
        # determine border margin
        margin_h = filter_height - 1
        margin_w = filter_width - 1
        data_set_x_border = data_set_x[:,:,(pic_h-margin_h):(2*pic_h+margin_h),(pic_w-margin_h):(2*pic_w+margin_w)]
        data_set_x = data_set_x[:,:,pic_h:(2*pic_h),pic_w:(2*pic_w)]

        print ">> After removing borders:"
        print "data_set_x_border: ", data_set_x_border.shape
        print "data_set_x:        ", data_set_x.shape

      print "     Compiling RBM..."

      rbm = morb.base.RBM()
      rbm.v = units.GaussianUnits(rbm, name='v')
      rbm.h = units.BinaryUnits(rbm, name='h')
      rbm.y = units.SoftmaxUnits(rbm, name='y')

      if args.convolution_type == "full":
        class DummyUnits(object):
          def __init__(self, name):
            self.name = name
            self.proxy_units = []
          def __repr__(self):
            return self.name
        rbm.v_border = DummyUnits(name="v border dummy")

        context_units = [rbm.v_border]
      else:
        context_units = []

      pmap = {
        "W":  theano.shared(value=prev_W, name="W"),
        "bv": theano.shared(value=prev_bv, name="bv"),
        "bh": theano.shared(value=prev_bh, name="bh"),
        "U":  theano.shared(value=prev_U, name="U"),
        "by": theano.shared(value=prev_by, name="by")
      }

      shape_info = {
        'hidden_maps': prev_W.shape[0],
        'visible_maps': prev_W.shape[1],
        'filter_height': prev_W.shape[2],
        'filter_width': prev_W.shape[3],
        'visible_height': pic_w,
        'visible_width': pic_w,
        'mb_size': 1
      }

      # parameters
      parameters.FixedBiasParameters(rbm, rbm.v.precision_units)
      if args.convolution_type == "full":
        rbm.W = borderconvparameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], 'W', name='W', shape_info=shape_info, var_fixed_border=rbm.v_border, alternative_gradient=True)
      elif args.convolution_type == "no":
        rbm.W = parameters.AdvancedProdParameters(rbm, [rbm.v, rbm.h], [3,1], 'W', name='W')
      elif args.convolution_type == "fullnoborder":
        rbm.W = borderconvparameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], 'W', name='W', shape_info=shape_info)

      # one bias per map (so shared across width and height):
      rbm.bv = parameters.SharedBiasParameters(rbm, rbm.v, 3, 2, 'bv', name='bv')

      if args.convolution_type == "no":
        rbm.bh = parameters.BiasParameters(rbm, rbm.h, 'bh', name='bh')
      else:
        rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 3, 2, 'bh', name='bh')

      # labels
      if args.convolution_type == "no":
        rbm.U = parameters.ProdParameters(rbm, [rbm.y, rbm.h], 'U', name='U')
      else:
        rbm.U = parameters.SharedProdParameters(rbm, [rbm.y, rbm.h], 3, 2, 'U', name='U', pooling_operator=T.sum)
      rbm.by = parameters.BiasParameters(rbm, rbm.y, 'by', name='by')

      initial_vmap = { rbm.v: T.tensor4('v'),
                       rbm.y: T.matrix('y') }
      if args.convolution_type == "full":
        initial_vmap[rbm.v_border] = T.tensor4('v border')

      # prediction
      predict = prediction.label_prediction(rbm, initial_vmap, pmap, \
          visible_units = [rbm.v], \
          label_units = [rbm.y], \
          hidden_units = [rbm.h],
          context_units = context_units,
          mb_size=1, mode=mode,
          logprob = False)

      print rbm


      print "     Computing predictions..."

      if args.convolution_type == "full":
        predicted_probs = np.concatenate([y for y, in predict({ rbm.v: data_set_x, rbm.v_border: data_set_x_border })])
      else:
        predicted_probs = np.concatenate([y for y, in predict({ rbm.v: data_set_x })])

      return predicted_probs


# classify training data,
# keep parameters
print
print "## TRAIN DATA"
train_predicted_label_probs = predict_labels(args, train_set_x)
# classify test data,
# reuse parameters
print
print "## TEST DATA"
test_predicted_label_probs = predict_labels(args, test_set_x)
print



train_predicted_labels = np.argmax(train_predicted_label_probs, axis=1) + 1
test_predicted_labels = np.argmax(test_predicted_label_probs, axis=1) + 1
train_y_true = sklearn.preprocessing.label_binarize(train_labels, range(1, args.n_states+1))
test_y_true = sklearn.preprocessing.label_binarize(test_labels, range(1, args.n_states+1))

if args.save_predictions:
  predictions = {}
  predictions["classes-train"] = range(1, args.n_states+1)
  predictions["true-labels-train"] = train_labels.reshape(-1)
  predictions["predicted-labels-train"] = train_predicted_labels
  predictions["predicted-probability-train"] = train_predicted_label_probs
  predictions["src-train"] = train_src
  predictions["classes-test"] = range(1, args.n_states+1)
  predictions["true-labels-test"] = test_labels.reshape(-1)
  predictions["predicted-labels-test"] = test_predicted_labels
  predictions["predicted-probability-test"] = test_predicted_label_probs
  predictions["src-test"] = test_src

  if args.save_predictions:
    print "Writing predictions to %s" % args.save_predictions
    np.savez_compressed(args.save_predictions, **predictions)

print "train accuracy:", sklearn.metrics.accuracy_score(train_labels, train_predicted_labels)
print "train confusion matrix:"
print sklearn.metrics.confusion_matrix(train_labels, train_predicted_labels)
print "train roc_auc_score:", sklearn.metrics.roc_auc_score(train_y_true, train_predicted_label_probs, average=None)
print
print "test accuracy:", sklearn.metrics.accuracy_score(test_labels, test_predicted_labels)
print "test confusion matrix:"
print sklearn.metrics.confusion_matrix(test_labels, test_predicted_labels)
print "test roc_auc_score:", sklearn.metrics.roc_auc_score(test_y_true, test_predicted_label_probs, average=None)
print


