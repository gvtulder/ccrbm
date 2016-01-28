# Loads a trained RBM and computes feature vectors.
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
parser.add_argument("--load-raw-feature-maps", action="store_true",
                    help="load raw feature maps from mat files")
parser.add_argument("--previous-layer", metavar="PKL", type=str, nargs="+")
parser.add_argument("--random-filters", action="store_true",
                    help="use random filters of PKL size")
parser.add_argument("--random-filters-seed", type=int,
                    help="generate random filters with this seed")
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
parser.add_argument("--skip-sigmoid", action="store_true")
parser.add_argument("--skip-normalisation", action="store_true")
parser.add_argument("--global-normalisation", action="store_true",
                    help="Use global normalisation, not per-patch")
parser.add_argument("--save-features", metavar="F", type=str,
                    help="write features to file",
                    default=None)
parser.add_argument("--subset", metavar="PROP", type=float,
                    help="Train and test on a small subset",
                    default=None)
parser.add_argument("--rng-seed", metavar="SEED", type=int, default=123)
parser.add_argument("--convolution-type", required=True,
                    choices=["no", "full", "fullnoborder"])
parser.add_argument("--pooling-approach", required=True,
                    choices=["histograms", "sum", "none"])
args = parser.parse_args()



print "load raw feat.: ", args.load_raw_feature_maps
print "previous layer: ", args.previous_layer
print "random filters: ", args.random_filters
print "random fil.seed:", args.random_filters_seed
print "skip normalis.: ", args.skip_normalisation
print "global normal.: ", args.global_normalisation
print "skip sigmoid:   ", args.skip_sigmoid
print "subsample:      ", args.subsample
print "subset:         ", args.subset
print "rng seed:       ", args.rng_seed
print "convolution:    ", args.convolution_type
print "pooling:        ", args.pooling_approach


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
  if args.load_raw_feature_maps:
    train_data.append(np.double(m['raw_cells']))
    train_labels.append(np.double(m['labels']))
  else:
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
  if args.load_raw_feature_maps:
    test_data.append(np.double(m['raw_cells']))
    test_labels.append(np.double(m['labels']))
  else:
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

if args.load_raw_feature_maps:

  ############################################
  # RAW FEATURE MAPS
  ############################################

  # convert to square feature maps
  pic_w_from_data = int(np.sqrt(train_data.shape[2]))
  train_set_x = train_data.reshape((train_data.shape[0], train_data.shape[1], pic_w_from_data, pic_w_from_data))
  test_set_x = test_data.reshape((test_data.shape[0], train_data.shape[1], pic_w_from_data, pic_w_from_data))

  # release
  train_data = None
  test_data = None
  

else:

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

def apply_convolution(args, data_set_x, random_filters={}, normalisation={}):
  ############################################
  # APPLY FIRST LAYER CONVO
  ############################################

  if args.random_filters:
    random_filters_rng = np.random.RandomState(args.random_filters_seed)

  if args.convolution_type == "fullnoborder":
    print "   Removing neighbourhoods:"
    print "     before:            ", data_set_x.shape

    # cut borders
    margin = (data_set_x.shape[2] - pic_w) / 2
    data_set_x = data_set_x[:,:,(margin):(pic_w + margin),(margin):(pic_w + margin)]

    print "     after:             ", data_set_x.shape

  for prev_layer in args.previous_layer:
    print ">> Processing layer: ", prev_layer

    data_set_x_conv_collect = []

    for fname in prev_layer.split(","):
      print "  >> Filter set:   ", fname
      with open(fname, "r") as f:
        prev_layer_params = cPickle.load(f)

        prev_W = prev_layer_params["W"]
        prev_bh = prev_layer_params["bh"]

        print "       prev_W.shape:  ", prev_W.shape
        print "       prev_bh.shape: ", prev_bh.shape

        if args.random_filters:
          print "     Replacing filters by random values..."
          if fname in random_filters:
            prev_W  = random_filters[fname]["prev_W"]
            prev_bh = random_filters[fname]["prev_bh"]
          else:
            prev_W = np.array(random_filters_rng.uniform(low=-1, high=1, size=prev_W.shape), dtype=prev_W.dtype)
            prev_bh = np.array(random_filters_rng.uniform(low=-1, high=1, size=prev_bh.shape), dtype=prev_bh.dtype)
            random_filters[fname] = { "prev_W": prev_W, "prev_bh": prev_bh }

        print "     Compiling convolution..."

        V = T.dtensor4()
        W = T.dtensor4()
        bh = T.dvector()
        W_flipped = W[:, :, ::-1, ::-1]
        subsample = (args.subsample, args.subsample)
        reshaped_bh = bh.dimshuffle('x',0,'x','x')
        if args.convolution_type == "fullnoborder":
          c = conv.conv2d(V, W_flipped, border_mode="valid", subsample=subsample)
        else:
          c = conv.conv2d(V, W_flipped, border_mode="full", subsample=subsample)
        c_act = activation_functions.sigmoid(c + reshaped_bh)
        if args.skip_sigmoid:
          conv_f = theano.function([V, W, bh], [ c ], on_unused_input="ignore")
        else:
          conv_f = theano.function([V, W, bh], [ c_act ])

        print "     Applying convolution..."

        start_time = time.time()
        data_set_x_conv = None
        n_samples = data_set_x.shape[0]
        batch_size = 5
        for i in xrange(0, n_samples, batch_size):
          cvf = conv_f(data_set_x[i:min(i+batch_size, n_samples), :,:,:], prev_W, prev_bh)[0]
          if data_set_x_conv is None:
            s = np.array(cvf.shape)
            s[0] = data_set_x.shape[0]
            data_set_x_conv = np.zeros(s, dtype=cvf.dtype)
          data_set_x_conv[i:min(i+batch_size, n_samples), :,:,:] = cvf
          if i % 10 == 0 and i > 0:
            print "     %d     %0.2f/s" % (i, float(i) / (time.time() - start_time))

        # release
        prev_layer_params = None
        prev_W = None
        prev_bh = None
        conv_f = None
        cvf = None

        print "     After this layer:"
        print "       data_set_x_conv:   ", data_set_x_conv.shape

        data_set_x_conv_collect.append(data_set_x_conv)

        # release
        data_set_x_conv = None

        # garbage collection
        gc.collect()

    if len(data_set_x_conv_collect) == 1:
      data_set_x = data_set_x_conv_collect[0]
    else:
      data_set_x = np.concatenate(data_set_x_conv_collect, axis=1)

    # release
    data_set_x_conv_collect = None

    # garbage collection
    gc.collect()

    print "   After this layer:"
    print "     data_set_x:        ", data_set_x.shape

    if args.global_normalisation:
      # normalise / whiten
      if prev_layer in normalisation:
        mu = normalisation[prev_layer]["mu"]
        sigma = normalisation[prev_layer]["sigma"]
      else:
        print "   Calculating normalisation parameters"
        mu = np.mean(data_set_x)
        sigma = np.std(data_set_x)
        normalisation[prev_layer] = { "mu": mu, "sigma": sigma }
      print "   Normalising layer output..."
      data_set_x -= mu
      data_set_x /= (0.25 * sigma)

    elif not args.skip_normalisation:
      # normalise / whiten
      print "   Normalising layer output..."
      n_samples = data_set_x.shape[0]
      data_set_rows = data_set_x.reshape(n_samples, -1)
      mu = np.mean(data_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
      sigma = memory_efficient_std(data_set_rows).reshape(n_samples, 1, 1, 1)
      # sigma = np.std(data_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
      data_set_x -= mu
      data_set_x /= (0.25 * sigma)
      # release
      data_set_rows = None

  # garbage collection
  gc.collect()


  # cut borders
  if args.convolution_type != "fullnoborder":
    margin = (data_set_x.shape[2] - pic_w) / 2
    data_set_x = data_set_x[:,:,(margin):(pic_w + margin),(margin):(pic_w + margin)]

    print "   After removing borders:"
    print "     data_set_x:        ", data_set_x.shape

  ############################################
  # END OF CONVOLUTION
  ############################################

  return (data_set_x, random_filters, normalisation)


def calculate_histograms(args, data_set_x, histogram_edges=None, least_common_multiple=24):
  # construct 'master histogram' with 48 bins
  # histogram (Lauge S\orensen)
  print ">> Master histogram..."
  if not histogram_edges:
    print "   Determining edges"
    histogram_edges = []
    for m in xrange(data_set_x.shape[1]):
      a = np.sort(data_set_x[:,m,:,:], axis=None)
      edges = a[ [int(i) for i in np.linspace(0, a.shape[0] - 1, num=least_common_multiple + 1)] ]
      edges[0]  = -np.inf
      edges[-1] = +np.inf
      histogram_edges.append(edges)

  hist_coll = []
  for i in xrange(data_set_x.shape[0]):
    hist_coll_i = []
    for m,edges in enumerate(histogram_edges):
      h, b = np.histogram(data_set_x[i,m,:,:], edges)
      h = np.cumsum(h).reshape([1,1,-1])
      hist_coll_i.append(h.astype(theano.config.floatX))
    hist_coll.append(np.concatenate(hist_coll_i, axis=1))
  data_set_x_conv = np.concatenate(hist_coll)

  data_set_x = data_set_x_conv

  print "   Histogram size:"
  print "     data_set_x:        ", data_set_x.shape

  # garbage collection
  gc.collect()

  return (data_set_x, histogram_edges)



if args.load_raw_feature_maps:
  # TODO
  raise "This might not work."



least_common_multiple = 24

# convolve training data,
# keep parameters
print
print "## TRAIN DATA"
(train_set_x, random_filters, normalisation) = \
    apply_convolution(args, train_set_x)
# convolve test data,
# reuse parameters
print
print "## TEST DATA"
(test_set_x, random_filters, normalisation) = \
    apply_convolution(args, test_set_x, random_filters, normalisation)
print


def save_features_to_file(dirname, \
       train_src, train_data, train_labels, \
       test_src, test_data, test_labels):
  # make feature vectors 1D
  this_train = train_data.reshape([train_data.shape[0],-1])
  this_test = test_data.reshape([test_data.shape[0],-1])

  # normalise feature vectors
  mu = np.mean(this_train, axis=0).reshape([1,-1])
  sigma = np.std(this_train, axis=0).reshape([1,-1])
  sigma[sigma == 0] = 0.5
  this_train -= mu
  this_train /= sigma
  this_test -= mu
  this_test /= sigma

  if not os.path.exists(dirname):
    os.makedirs(dirname)

  # write data
  np.savez_compressed("%s/data.train.npz" % dirname,
                      samples=this_train,
                      labels=train_labels.squeeze(),
                      order=train_src)
  np.savez_compressed("%s/data.test.npz" % dirname,
                      samples=this_test,
                      labels=test_labels.squeeze(),
                      order=test_src)



if args.convolution_type == "full":
  conv_type_str = "conv"
elif args.convolution_type == "fullnoborder":
  conv_type_str = "conv-noborder"
else:
  conv_type_str = "noconv"

if args.pooling_approach == "histograms":
  print ">> Calculating main histogram"
  (train_set_x, histogram_edges) = \
      calculate_histograms(args, train_set_x, least_common_multiple=least_common_multiple)
  (test_set_x, histogram_edges) = \
      calculate_histograms(args, test_set_x, histogram_edges, least_common_multiple=least_common_multiple)
  print

  # generate features at multiple levels
  print ">> Saving with histogram bins:"
  for histogram_bins in (2,3,4,6,8,12):
    print "   %d" % histogram_bins

    # use histogram bins at this interval
    step_size = least_common_multiple / histogram_bins
    this_train = train_set_x[:, :, range(step_size - 1, least_common_multiple, step_size)]
    for i in xrange(histogram_bins-1, 0, -1):
      this_train[:,:,i] -= this_train[:,:,i-1]
    this_test  = test_set_x[:, :, range(step_size - 1, least_common_multiple, step_size)]
    for i in xrange(histogram_bins-1, 0, -1):
      this_test[:,:,i] -= this_test[:,:,i-1]

    save_features_to_file("%s/%s.bins-%d" % (args.save_features, conv_type_str, histogram_bins), \
      train_src, this_train, train_labels, \
      test_src, this_test, test_labels)

elif args.pooling_approach == "sum":
  print ">> Sum per feature map"
  train_set_x = np.sum(train_set_x, (2,3))
  test_set_x  = np.sum(test_set_x, (2,3))
  # save features
  save_features_to_file("%s/%s.sums" % (args.save_features, conv_type_str), \
    train_src, train_set_x, train_labels, \
    test_src, test_set_x, test_labels)

else:
  print ">> Save RBM representation as features"
  # save features (postprocessing already done)
  save_features_to_file("%s/%s" % (args.save_features, conv_type_str), \
    train_src, train_set_x, train_labels, \
    test_src, test_set_x, test_labels)


print "Done."
print

