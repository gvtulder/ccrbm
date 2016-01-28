# Trains an RBM.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy.io as sio
from scipy import ndimage
import scipy.misc
import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters, prediction, objectives
import theano
import theano.tensor as T
import numpy as np
import gzip, cPickle, time
import json, sys, os, time, os.path, math
import re
import gc
from utils import one_hot

from plot_filters import plot_filters, filter_dot_distance
from confmat import Confmat

import borderconvparameters

from theano import ProfileMode
mode = None

theano.config.floatX = 'float32'

# do not use scientific notation
np.set_printoptions(suppress=True)

# mode = theano.compile.DebugMode(require_matching_strides=False)
theano.config.exception_verbosity = 'high'


############################################
# SETTINGS
############################################

visible_maps = 1
hidden_maps = 32 # 8*8 # 100 # 50
filter_height = 5 # pic_w
filter_width = 5 # pic_w
offsets = filter_height


import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("--experiment-id", metavar="ID", help="experiment ID")
parser.add_argument("--previous-layer", metavar="PKL", type=str, nargs="+")
parser.add_argument("--subsample", metavar="PX", type=int,
                    help="subsample first layer",
                    default=1)
parser.add_argument("--previous-layer-binary", action="store_true",
                    help="previous layer output as binary")
parser.add_argument("--beta", metavar="BETA", type=float,
                    help="proportion of generative learning",
                    default=0.9)
parser.add_argument("--beta-decay", metavar="BETA_DECAY", type=float,
                    help="actual beta = beta * (decay**epoch)",
                    default=1)
parser.add_argument("--learning-rate", metavar="RATE", type=float,
                    help="learning rate",
                    default=0.0001)
parser.add_argument("--learning-rate-decay-start", metavar="EPOCH", type=int,
                    help="learning rate decay, decay start")
parser.add_argument("--learning-rate-decay-end", metavar="EPOCH", type=int,
                    help="learning rate decay, decay end")
parser.add_argument("--learning-rate-decay-end-rate", metavar="RATE", type=float,
                    help="learning rate decay, final rate")
parser.add_argument("--learning-rate-bias", metavar="RATE", type=float,
                    help="learning rate for the bias")
parser.add_argument("--target-sparsity", metavar="ACTIVATION", type=float,
                    help="desired mean activation per feature map",
                    default=None)
parser.add_argument("--sparsity-lambda", metavar="LAMBDA", type=float,
                    help="lambda parameter",
                    default=None)
parser.add_argument("--goh-target-sparsity", metavar="ACTIVATION", type=float,
                    help="desired mean activation per feature map",
                    default=None)
parser.add_argument("--goh-sparsity-lambda", metavar="LAMBDA", type=float,
                    help="lambda parameter",
                    default=None)
parser.add_argument("--fixed-hidden-bias", action="store_true")
parser.add_argument("--non-shared-bias", action="store_true")
parser.add_argument("--no-U-pooling", action="store_true")
parser.add_argument("--multiple-hidden-bias", type=int,
                    help="number of hidden biases per filter",
                    default=None)
parser.add_argument("--bias-decay", metavar="BIAS_DECAY", type=float,
                    help="hidden bias -= BIAS_DECAY",
                    default=None)
parser.add_argument("--hidden-maps", metavar="N", type=int,
                    help="number of hidden maps",
                    default=hidden_maps)
parser.add_argument("--filter-height", metavar="H", type=int,
                    help="filter height",
                    default=filter_height)
parser.add_argument("--filter-width", metavar="W", type=int,
                    help="filter width",
                    default=filter_width)
parser.add_argument("--epochs", metavar="N", type=int,
                    help="epochs",
                    default=3000)
parser.add_argument("--train-scans", metavar="S", type=str,
                    help="scans used for training",
                    default="069,048")
parser.add_argument("--test-scans", metavar="S", type=str,
                    help="scans used for testing",
                    default="002,007")
parser.add_argument("--only-class", metavar="C", type=int,
                    help="train/test for one class",
                    default=None)
parser.add_argument("--n-states", metavar="N", type=int,
                    help="number of classes",
                    default=4)
parser.add_argument("--mb-size", metavar="MB", type=int, default=10)
parser.add_argument("--image-size", metavar="PX", type=int)
parser.add_argument("--image-width", metavar="PX", type=int)
parser.add_argument("--image-height", metavar="PX", type=int)
parser.add_argument("--subpatches", metavar="PX", type=int)
parser.add_argument("--convolution-type", required=True,
                    choices=["no","full","fullnoborder"])
parser.add_argument("--gaussian-blur", metavar="SIGMA", type=float,
                    help="Gaussian blur with sigma",
                  default=None)
parser.add_argument("--zca-whiten", action="store_true",
                    help="Apply ZCA whitening",
                    default=None)
parser.add_argument("--global-normalisation", action="store_true",
                    help="Use global normalisation, not per-patch")
parser.add_argument("--k-train", metavar="N", type=int,
                    help="CD steps for training",
                    default=1)
parser.add_argument("--k-eval", metavar="N", type=int,
                    help="CD steps for evaluation",
                    default=1)
parser.add_argument("--subset", metavar="PROP", type=float,
                    help="Train and test on a small subset",
                    default=None)
parser.add_argument("--subset-noshuffle", metavar="PROP", type=float,
                    help="Train and test on the first PROP of samples",
                    default=None)
parser.add_argument("--train-label-only-discriminative",
                    nargs="?", type=float,
                    const=-1000, default=None)
parser.add_argument("--ignore-labels", action="store_true",
                    help="Label units have no effects on filter learning")
parser.add_argument("--weight-w-init-std", type=float, default=None)
parser.add_argument("--weight-u-init-std", type=float, default=None)
parser.add_argument("--evaluate-every", metavar="EPOCHS", type=int, default=10)
parser.add_argument("--plot-every", metavar="EPOCHS", type=int, default=20)
parser.add_argument("--test-every", metavar="EPOCHS", type=int, default=100)
parser.add_argument("--resume-run", metavar="RUN_ID", type=str)
parser.add_argument("--rng-seed", metavar="SEED", type=int, default=123)
args = parser.parse_args()

hidden_maps = args.hidden_maps
filter_height = args.filter_height
filter_width = args.filter_width
offsets = filter_height

epochs = args.epochs
learning_rate = args.learning_rate

if args.image_size:
  pic_w = args.image_size
  pic_h = args.image_size
else:
  pic_w = 15
  pic_h = 15
if args.image_width:
  pic_w = args.image_width
if args.image_height:
  pic_h = args.image_height
mb_size = args.mb_size
n_states = args.n_states

if args.convolution_type == "no":
  pic_h = filter_height
  pic_w = filter_width

shape_info = {
  'hidden_maps': hidden_maps,
  'visible_maps': visible_maps,
  'filter_height': filter_height,
  'filter_width': filter_width,
  'visible_height': pic_h,
  'visible_width': pic_w,
  'mb_size': mb_size
}

pooling_operator = T.sum

if args.train_label_only_discriminative and args.train_label_only_discriminative < 0:
  args.train_label_only_discriminative = args.learning_rate


run_id = "%s-%s-%d" % (args.experiment_id, time.strftime("%Y%m%d-%H%M%S"), os.getpid())
if args.only_class:
  run_id = "%s-class%d" % (run_id, args.only_class)
print "run: %s" % run_id
print "experiment: %s" % args.experiment_id
print

print "previous layer: ", args.previous_layer
print "subsample:      ", args.subsample
print "previous layer binary: ", args.previous_layer_binary
print "shape info:   ", shape_info
print "beta:         ", args.beta
print "beta decay:   ", args.beta_decay
print "pooling oper: ", pooling_operator
print "convolution:  ", args.convolution_type
print "subpatches:   ", args.subpatches
print "gaussian blur:", args.gaussian_blur
print "zca whiten:   ", args.zca_whiten
print "learning rate:", args.learning_rate
print "  decay start:", args.learning_rate_decay_start
print "  decay end:  ", args.learning_rate_decay_end
print "  final rate: ", args.learning_rate_decay_end_rate
print "  for bias:   ", args.learning_rate_bias
print "discrim. lrat:", args.train_label_only_discriminative
print "sparsity:     ", args.target_sparsity
print "sparsity lamb:", args.sparsity_lambda
print "goh sparsity: ", args.goh_target_sparsity
print "goh sparslamb:", args.goh_sparsity_lambda
print "fix hidd.bias:", args.fixed_hidden_bias
print "non-shared bias:", args.non_shared_bias
print "no U-pooling: ", args.no_U_pooling
print "multi hidbias:", args.multiple_hidden_bias
print "bias decay:   ", args.bias_decay
print "mb_size:      ", mb_size
print "epochs:       ", epochs
print "only class:   ", args.only_class
print "subset:       ", args.subset
print "subset noshuf:", args.subset_noshuffle
print "ignore labels:", args.ignore_labels
print "weight W init std: ", args.weight_w_init_std
print "weight U init std: ", args.weight_u_init_std
print "rng seed:     ", args.rng_seed


numpy_rng = np.random.RandomState(args.rng_seed)


############################################
# DATA
############################################

# load data
print ">> Loading dataset..."

# scans with 15x15 pixels

scans = ["002","007","032","048","069","072","075","087","108","109","112","114","137","139","156","175","188","189","190","201","205","224","233","250","272","279","281","283","306","312","318","320","326","329","351","362","363","367","373","388","430"]
scans = ["002","007","032","048","069","072","075","087","108","109","112","114","137"]
scans = ["002","007","032","048"] # ,"069","072","075","087","108","109","112","114","137"]

# test_scans = sys.argv[1:]
# train_scans = [s for s in scans if not s in test_scans]
# train_scans = train_scans[0:2]

# test_scans = ["002","007"]
# train_scans = ["069","048"]

train_scans = args.train_scans.split(",")
test_scans = args.test_scans.split(",")

print "train: "+(",".join(train_scans))
print "test:  "+(",".join(test_scans))
print

if "032" in test_scans or "032" in train_scans:
  raise Exception("032 not okay.")

# load train scans
train_data = []
train_labels = []
for s in train_scans:
  if ".mat" in s:
    m = sio.loadmat(s)
  elif ".jpg" in s:
    s_label, s_filename = s.split(":")
    m = scipy.misc.imread(s_filename)
    train_data.append(np.expand_dims(m,0).astype(theano.config.floatX))
    train_labels.append(np.reshape(np.double(s_label),[1,1]).astype(theano.config.floatX))
    pic_h = m.shape[0]
    pic_w = m.shape[1]
  elif args.convolution_type == "full" or args.previous_layer:
    m = sio.loadmat("SALD-cells-with-borders/"+s+".mat")
  else:
    m = sio.loadmat("SALD-cells/"+s+".mat")
  if ".jpg" in s:
    pass
  elif 'image' in m:
    train_data.append(m['image'].astype(theano.config.floatX))
  elif args.convolution_type == "full" or args.previous_layer:
    train_data.append(np.transpose(np.double(m['neighbourhoods']).astype(theano.config.floatX)))
  else:
    train_data.append(np.transpose(np.double(m['cells']).astype(theano.config.floatX)))
  if not ".jpg" in s:
    train_labels.append(np.transpose(m['labels'].astype(theano.config.floatX)))
  m = None
train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)
if args.only_class:
  subset = (train_labels[:,0] == args.only_class)
  train_data = train_data[subset]
  train_labels = train_labels[subset]

# remove any extra classes
subset = (train_labels[:,0] <= args.n_states)
train_data = train_data[subset]
train_labels = train_labels[subset]

if args.subset_noshuffle:
  n = min(train_data.shape[0], int(args.subset_noshuffle * n))
  train_data = train_data[0:(np.floor_divide(n, mb_size) * mb_size)]
  train_labels = train_labels[0:(np.floor_divide(n, mb_size) * mb_size)]

order = numpy_rng.permutation(train_data.shape[0])
n = train_data.shape[0]
if not args.subset is None:
  n = min(n, int(args.subset * n))
order = order[0:(np.floor_divide(n, mb_size) * mb_size)]
train_data = train_data[order]
train_labels = train_labels[order]

print "train shape:", train_data.shape
print "train shape:", train_labels.shape

# load test scans
test_data = []
test_labels = []
for s in test_scans:
  if ".mat" in s:
    m = sio.loadmat(s)
  elif ".jpg" in s:
    s_label, s_filename = s.split(":")
    m = scipy.misc.imread(s_filename)
    test_data.append(np.expand_dims(m,0).astype(theano.config.floatX))
    test_labels.append(np.reshape(np.double(s_label),[1,1]).astype(theano.config.floatX))
  elif args.convolution_type == "full" or args.previous_layer:
    m = sio.loadmat("SALD-cells-with-borders/"+s+".mat")
  else:
    m = sio.loadmat("SALD-cells/"+s+".mat")
  if ".jpg" in s:
    pass
  elif 'image' in m:
    test_data.append(np.double(m['image']).astype(theano.config.floatX))
  elif args.convolution_type == "full" or args.previous_layer:
    test_data.append(np.transpose(np.double(m['neighbourhoods']).astype(theano.config.floatX)))
  else:
    test_data.append(np.transpose(np.double(m['cells']).astype(theano.config.floatX)))
  if not ".jpg" in s:
    test_labels.append(np.transpose(m['labels'].astype(theano.config.floatX)))
  m = None
test_data = np.concatenate(test_data)
test_labels = np.concatenate(test_labels)
if args.only_class:
  subset = (test_labels[:,0] == args.only_class)
  test_data = test_data[subset]
  test_labels = test_labels[subset]

# remove any extra classes
subset = (test_labels[:,0] <= args.n_states)
test_data = test_data[subset]
test_labels = test_labels[subset]

if args.subset_noshuffle:
  n = min(test_data.shape[0], int(args.subset_noshuffle * n))
  test_data = test_data[0:(np.floor_divide(n, mb_size) * mb_size)]
  test_labels = test_labels[0:(np.floor_divide(n, mb_size) * mb_size)]

order = numpy_rng.permutation(test_data.shape[0])
n = test_data.shape[0]
if not args.subset is None:
  n = min(n, int(args.subset * n))
order = order[0:(np.floor_divide(n, mb_size) * mb_size)]
test_data = test_data[order]
test_labels = test_labels[order]

print "test shape: ", test_data.shape
print "test shape: ", test_labels.shape
print

train_distr = [ sum(sum(train_labels==i)) for i in np.sort(np.unique(train_labels)) ]
test_distr  = [ sum(sum(test_labels==i)) for i in np.sort(np.unique(test_labels)) ]
print "train distribution: ", train_distr
print "train distribution: ", np.asarray(train_distr, dtype=float) / sum(train_distr)
print "test distribution:  ", test_distr
print "test distribution:  ", np.asarray(test_distr, dtype=float) / sum(test_distr)
print

# garbage collection
gc.collect()



############################################
# CONVERT TO INPUTS
############################################

if train_data.ndim == 2:
  pic_h_from_data = int(np.sqrt(train_data.shape[1]))
  pic_w_from_data = int(np.sqrt(train_data.shape[1]))
  train_set_x = train_data.reshape((train_data.shape[0], 1, pic_h_from_data, pic_w_from_data))
else:
  train_set_x = train_data.reshape([train_data.shape[0], 1] + list(train_data.shape[1:100]))
pic_h_from_data = int(train_set_x.shape[2])
pic_w_from_data = int(train_set_x.shape[3])
if test_data.ndim == 2:
  test_set_x = test_data.reshape((test_data.shape[0], 1, pic_h_from_data, pic_w_from_data))
else:
  test_set_x = test_data.reshape([test_data.shape[0], 1] + list(test_data.shape[1:100]))

if args.subpatches:
  print ">> Extracting subpatches"
  print "   Before:      train  ", train_set_x.shape
  print "                test   ", test_set_x.shape
  # create subpatches
  cells_sub = []
  labels_sub = []
  for i in xrange(train_set_x.shape[0]):
    for j in xrange(args.subpatches):
      x = numpy_rng.randint(0, pic_w_from_data - pic_w)
      y = numpy_rng.randint(0, pic_h_from_data - pic_h)
      cells_sub.append(train_set_x[i:(i+1), :, y:(y+pic_h), x:(x+pic_w)])
      labels_sub.append(train_labels[i:(i+1), :])
  train_set_x = np.concatenate(cells_sub)
  train_labels = np.concatenate(labels_sub)

  order = numpy_rng.permutation(train_set_x.shape[0])
  train_set_x = train_set_x[order]
  train_labels = train_labels[order]

  cells_sub = []
  labels_sub = []
  for i in xrange(test_set_x.shape[0]):
    for j in xrange(args.subpatches):
      x = numpy_rng.randint(0, pic_w_from_data - pic_w)
      y = numpy_rng.randint(0, pic_h_from_data - pic_h)
      cells_sub.append(test_set_x[i:(i+1), :, y:(y+pic_h), x:(x+pic_w)])
      labels_sub.append(test_labels[i:(i+1), :])
  test_set_x = np.concatenate(cells_sub)
  test_labels = np.concatenate(labels_sub)

  order = numpy_rng.permutation(test_set_x.shape[0])
  test_set_x = test_set_x[order]
  test_labels = test_labels[order]

  cells_sub = None
  labels_sub = None

  print "   After:       train  ", train_set_x.shape
  print "                test   ", test_set_x.shape

  pic_h_from_data = pic_h
  pic_w_from_data = pic_w

if not args.convolution_type == "full" and not args.previous_layer:
  assert pic_h == pic_h_from_data
  assert pic_w == pic_w_from_data

# release
train_data = None
test_data = None

train_set_y = one_hot(train_labels[0:train_set_x.shape[0],:] - 1, n_states)
test_set_y = one_hot(test_labels[0:test_set_x.shape[0],:] - 1, n_states)

if args.gaussian_blur:
  print ">> Applying Gaussian blur"
  for i in xrange(0, train_set_x.shape[0]):
    for j in xrange(0, train_set_x.shape[1]):
      train_set_x[i,j,:,:] = ndimage.gaussian_filter(train_set_x[i,j,:,:], sigma=args.gaussian_blur)
  for i in xrange(0, test_set_x.shape[0]):
    for j in xrange(0, test_set_x.shape[1]):
      test_set_x[i,j,:,:] = ndimage.gaussian_filter(test_set_x[i,j,:,:], sigma=args.gaussian_blur)

if args.zca_whiten:
  # http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening
  import scipy.linalg
  print ">> ZCA whitening"
  n_samples = train_set_x.shape[0]
  train_set_rows = train_set_x.reshape(n_samples, -1)
  mu = np.mean(train_set_rows, axis=1).reshape(n_samples, 1)
  train_set_rows -= mu

  sigma = np.dot(np.transpose(train_set_rows), train_set_rows) / n_samples
  U, s, Vh = scipy.linalg.svd(sigma)
  k = np.where(np.cumsum(s) / np.sum(s) > 0.90)[0][0] # explain 90% of variance
  epsilon = 0.1

  xZCAwhite = np.dot(np.dot(np.dot(U, np.diag(1.0 / np.sqrt(s + epsilon))), np.transpose(U)), np.transpose(train_set_rows))
  train_set_x = np.transpose(xZCAwhite).reshape(n_samples, 1, pic_h_from_data, pic_w_from_data)

  n_samples = test_set_x.shape[0]
  test_set_rows = test_set_x.reshape(n_samples, -1)
  mu = np.mean(test_set_rows, axis=1).reshape(n_samples, 1)
  test_set_rows -= mu
  xZCAwhite = np.dot(np.dot(np.dot(U, np.diag(1.0 / np.sqrt(s + epsilon))), np.transpose(U)), np.transpose(test_set_rows))
  test_set_x = np.transpose(xZCAwhite).reshape(n_samples, 1, pic_h_from_data, pic_w_from_data)

  xZCAwhite = None

if args.global_normalisation:
  # normalise / whiten
  global_mu = np.mean(train_set_x)
  global_sigma = np.std(train_set_x)
  train_set_x -= global_mu
  train_set_x /= (0.25 * global_sigma)
  test_set_x -= global_mu
  test_set_x /= (0.25 * global_sigma)

else:
  # normalise / whiten
  print ">> Normalising training data..."
  n_samples = train_set_x.shape[0]
  # pic_h = train_set_x.shape[2]
  # pic_w = train_set_x.shape[3]
  train_set_rows = train_set_x.reshape(n_samples, -1)
  mu = np.mean(train_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
  sigma = np.std(train_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
  train_set_x -= mu
  train_set_x /= (0.25 * sigma)
  # release
  train_set_rows = None

  print ">> Normalising testing data..."
  n_samples = test_set_x.shape[0]
  # pic_h = test_set_x.shape[2]
  # pic_w = test_set_x.shape[3]
  test_set_rows = test_set_x.reshape(n_samples, -1)
  mu = np.mean(test_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
  sigma = np.std(test_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
  test_set_x -= mu
  test_set_x /= (0.25 * sigma)
  # release
  test_set_rows = None

if not train_set_x.dtype == theano.config.floatX:
  train_set_x = train_set_x.astype(theano.config.floatX)
if not test_set_x.dtype == theano.config.floatX:
  test_set_x = test_set_x.astype(theano.config.floatX)

if args.convolution_type == "full" or args.previous_layer:
  # determine border margin
  margin_h = filter_height - 1
  margin_w = filter_width - 1
  train_set_x_border = train_set_x[:,:,(pic_h-margin_h):(2*pic_h+margin_h),(pic_w-margin_h):(2*pic_w+margin_w)]
  test_set_x_border = test_set_x[:,:,(pic_h-margin_h):(2*pic_h+margin_h),(pic_w-margin_h):(2*pic_w+margin_w)]
  train_set_x = train_set_x[:,:,pic_h:(2*pic_h),pic_w:(2*pic_w)]
  test_set_x = test_set_x[:,:,pic_h:(2*pic_h),pic_w:(2*pic_w)]

  print ">> After removing borders:"
  print "train_set_x_border: ", train_set_x_border.shape
  print "test_set_x_border:  ", test_set_x_border.shape
  print "train_set_x:        ", train_set_x.shape
  print "test_set_x:         ", test_set_x.shape

# garbage collection
gc.collect()


############################################
# APPLY FIRST LAYER CONVO
############################################

if args.previous_layer:
  from morb import activation_functions
  from theano.tensor.nnet import conv

  for prev_layer in args.previous_layer:
    print ">> Processing layer: ", prev_layer

    train_set_x_conv_border_collect = []
    test_set_x_conv_border_collect = []

    for fname in prev_layer.split(","):
      print "  >> Filter set:   ", fname
      with open(fname, "r") as f:
        prev_layer_params = cPickle.load(f)

        prev_W = prev_layer_params["W"]
        prev_bh = prev_layer_params["bh"]

        print "       prev_W.shape:  ", prev_W.shape
        print "       prev_bh.shape: ", prev_bh.shape

        if args.convolution_type == "no":
          # TODO
          print "TODO"
          sys.exit()

        print "     Applying convolution..."

        V = T.dtensor4()
        W = T.dtensor4()
        bh = T.dvector()
        W_flipped = W[:, :, ::-1, ::-1]
        subsample = (args.subsample, args.subsample)
        reshaped_bh = bh.dimshuffle('x',0,'x','x')
        c = conv.conv2d(V, W_flipped, border_mode="full", subsample=subsample)
        c_act = activation_functions.sigmoid(c + reshaped_bh)
        conv_f = theano.function([V, W, bh], [ c_act ])

        train_set_x_conv = []
        for i in xrange(0, train_set_x.shape[0]):
          cvf = conv_f(train_set_x_border[i:i+1, :,:,:], prev_W, prev_bh)[0]
          if not cvf.dtype == theano.config.floatX:
            cvf = cvf.astype(theano.config.floatX)
          train_set_x_conv.append(cvf)
        train_set_x_conv = np.concatenate(train_set_x_conv)

        test_set_x_conv = []
        for i in xrange(0, test_set_x.shape[0]):
          cvf = conv_f(test_set_x_border[i:i+1, :,:,:], prev_W, prev_bh)[0]
          if not cvf.dtype == theano.config.floatX:
            cvf = cvf.astype(theano.config.floatX)
          test_set_x_conv.append(cvf)
        test_set_x_conv = np.concatenate(test_set_x_conv)

        # release
        prev_layer_params = None
        prev_W = None
        prev_bh = None
        conv_f = None
        cvf = None

        print "     After this layer:"
        print "       train_set_x_conv:   ", train_set_x_conv.shape
        print "       test_set_x_conv:    ", test_set_x_conv.shape

        # determine border margin
        margin_h = (train_set_x_conv.shape[2] - train_set_x_border.shape[2]) / 2
        margin_w = (train_set_x_conv.shape[3] - train_set_x_border.shape[3]) / 2
        train_set_x_conv_border = train_set_x_conv[:,:,(margin_h):(train_set_x_conv.shape[2]-margin_h),(margin_w):(train_set_x_conv.shape[3]-margin_w)]
        test_set_x_conv_border = test_set_x_conv[:,:,(margin_h):(test_set_x_conv.shape[2]-margin_h),(margin_w):(test_set_x_conv.shape[3]-margin_w)]

        if not train_set_x_conv_border.dtype == theano.config.floatX:
          train_set_x_conv_border = train_set_x_conv_border.astype(theano.config.floatX)
        if not test_set_x_conv_border.dtype == theano.config.floatX:
          test_set_x_conv_border = test_set_x_conv_border.astype(theano.config.floatX)

        train_set_x_conv_border_collect.append(train_set_x_conv_border)
        test_set_x_conv_border_collect.append(test_set_x_conv_border)

        # release
        train_set_x_conv_border = None
        test_set_x_conv_border = None

    train_set_x_border = np.concatenate(train_set_x_conv_border_collect, axis=1)
    test_set_x_border = np.concatenate(test_set_x_conv_border_collect, axis=1)

    # release
    train_set_x_conv_border_collect = None
    test_set_x_conv_border_collect = None

    print "   After this layer:"
    print "     train_set_x_border: ", train_set_x_border.shape
    print "     test_set_x_border:  ", test_set_x_border.shape

    # remove borders
    margin_h = (train_set_x_border.shape[2] - pic_h) / 2
    margin_w = (train_set_x_border.shape[3] - pic_w) / 2
    train_set_x = train_set_x_border[:,:,(margin_h):(pic_h + margin_h),(margin_w):(pic_w + margin_w)]
    test_set_x = test_set_x_border[:,:,(margin_h):(pic_h + margin_h),(margin_w):(pic_w + margin_w)]

    print "   After removing borders:"
    print "     train_set_x:        ", train_set_x.shape
    print "     test_set_x:         ", test_set_x.shape

    visible_maps = train_set_x.shape[1]
    shape_info['visible_maps'] = visible_maps

    # normalise / whiten
    if not args.previous_layer_binary:
      if args.global_normalisation:
        # normalise / whiten
        mu = np.mean(train_set_x)
        sigma = np.std(train_set_x)
        train_set_x_border -= mu
        train_set_x_border /= (0.25 * sigma)
        test_set_x_border -= mu
        test_set_x_border /= (0.25 * sigma)

      else:
        print "   Normalising layer output (training data)..."
        n_samples = train_set_x.shape[0]
        # pic_h = train_set_x.shape[2]
        # pic_w = train_set_x.shape[3]
        train_set_rows = train_set_x.reshape(n_samples, -1)
        mu = np.mean(train_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
        sigma = np.std(train_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
        train_set_x_border -= mu
        train_set_x_border /= (0.25 * sigma)
        # release
        train_set_rows = None

        print "   Normalising layer output (testing data)..."
        n_samples = test_set_x.shape[0]
        # pic_h = test_set_x.shape[2]
        # pic_w = test_set_x.shape[3]
        test_set_rows = test_set_x.reshape(n_samples, -1)
        mu = np.mean(test_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
        sigma = np.std(test_set_rows, axis=1).reshape(n_samples, 1, 1, 1)
        test_set_x_border -= mu
        test_set_x_border /= (0.25 * sigma)
        # release
        test_set_rows = None

      if not train_set_x_border.dtype == theano.config.floatX:
        train_set_x_border = train_set_x_border.astype(theano.config.floatX)
      if not test_set_x_border.dtype == theano.config.floatX:
        test_set_x_border = test_set_x_border.astype(theano.config.floatX)

      # remove borders again,
      # so numpy can make train_set_x a view of train_set_x_border
      margin_h = (train_set_x_border.shape[2] - pic_w) / 2
      margin_w = (train_set_x_border.shape[3] - pic_w) / 2
      train_set_x = train_set_x_border[:,:,(margin_h):(pic_h + margin_h),(margin_w):(pic_w + margin_w)]
      test_set_x = test_set_x_border[:,:,(margin_h):(pic_h + margin_h),(margin_w):(pic_w + margin_w)]

      print "   After removing borders:"
      print "     train_set_x:        ", train_set_x.shape
      print "     test_set_x:         ", test_set_x.shape

# garbage collection
gc.collect()


############################################
# CONSTRUCT RBM
############################################

print ">> Constructing RBM..."
fan_in = visible_maps * filter_height * filter_width
if args.weight_w_init_std:
  weight_w_std = args.weight_w_init_std
else:
  weight_w_std = 0.5 / np.sqrt(fan_in)
if args.weight_u_init_std:
  weight_u_std = args.weight_u_init_std
else:
  weight_u_std = 4*np.sqrt(6./(hidden_maps+n_states))

# initial values
if args.convolution_type == "no":
  initial_W = np.asarray(
              numpy_rng.normal(
                  0, weight_w_std,
                  size = (visible_maps, pic_h, pic_w, hidden_maps)
              ), dtype=theano.config.floatX)
  initial_U = np.asarray(
              numpy_rng.uniform(
                  low   = -weight_u_std,
                  high  =  weight_u_std,
                  size  =  (n_states, hidden_maps)
              ), dtype=theano.config.floatX)
  if args.no_U_pooling:
    raise "No U pooling not supported with no convolution"
elif args.multiple_hidden_bias:
  initial_W = np.asarray(
              numpy_rng.normal(
                  0, weight_w_std,
                  size = (hidden_maps, visible_maps, filter_height, filter_width)
              ), dtype=theano.config.floatX)
  initial_U = np.asarray(
              numpy_rng.uniform(
                  low   = -weight_u_std,
                  high  =  weight_u_std,
                  size  =  (n_states, hidden_maps, args.multiple_hidden_bias)
              ), dtype=theano.config.floatX)
  if args.no_U_pooling:
    raise "No U pooling not supported with multiple hidden bias"
else:
  initial_W = np.asarray(
              numpy_rng.normal(
                  0, weight_w_std,
                  size = (hidden_maps, visible_maps, filter_height, filter_width)
              ), dtype=theano.config.floatX)
  if args.no_U_pooling:
    margin_h = shape_info['filter_height'] - 1
    margin_w = shape_info['filter_width'] - 1
    hidden_height = shape_info['visible_height'] + (2 * margin_h) - shape_info['filter_height'] + 1
    hidden_width = shape_info['visible_width'] + (2 * margin_w) - shape_info['filter_width'] + 1
    initial_U = np.asarray(
                numpy_rng.uniform(
                    low   = -weight_u_std,
                    high  =  weight_u_std,
                    size  =  (n_states, hidden_maps, hidden_height, hidden_width)
                ), dtype=theano.config.floatX)
  else:
    initial_U = np.asarray(
                numpy_rng.uniform(
                    low   = -weight_u_std,
                    high  =  weight_u_std,
                    size  =  (n_states, hidden_maps)
                ), dtype=theano.config.floatX)
initial_bv = np.zeros(visible_maps, dtype = theano.config.floatX)
initial_by = np.zeros(n_states, dtype = theano.config.floatX)
if args.non_shared_bias:
  if not args.convolution_type == "full":
    raise "Unsupported: non-shared bias needs full border convolution"
  margin_h = shape_info['filter_height'] - 1
  margin_w = shape_info['filter_width'] - 1
  hidden_height = shape_info['visible_height'] + (2 * margin_h) - shape_info['filter_height'] + 1
  hidden_width = shape_info['visible_width'] + (2 * margin_w) - shape_info['filter_width'] + 1
  initial_bh = np.zeros((hidden_maps, hidden_height, hidden_width), dtype = theano.config.floatX)
elif args.multiple_hidden_bias:
  initial_bh = np.zeros((hidden_maps, args.multiple_hidden_bias), dtype = theano.config.floatX)
else:
  initial_bh = np.zeros(hidden_maps, dtype = theano.config.floatX)

# units
# rbms.SigmoidBinaryRBM(n_visible, n_hidden)
rbm = morb.base.RBM()
if args.previous_layer_binary:
  rbm.v = units.BinaryUnits(rbm, name='v') # visibles
else:
  rbm.v = units.GaussianUnits(rbm, name='v') # visibles
# rbm.v = units.BinaryUnits(rbm, name='v') # visibles
rbm.h = units.BinaryUnits(rbm, name='h') # hiddens
# rbm.y = units.BinaryUnits(rbm, name='y')
if not args.ignore_labels:
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
  "W":  theano.shared(value=initial_W, name="W"),
  "bv": theano.shared(value=initial_bv, name="bv"),
  "bh": theano.shared(value=initial_bh, name="bh")
}
if not args.ignore_labels:
  pmap["U"] =theano.shared(value=initial_U, name="U")
  pmap["by"] =theano.shared(value=initial_by, name="by")

# parameters
if not args.previous_layer_binary:
  parameters.FixedBiasParameters(rbm, rbm.v.precision_units)
if args.multiple_hidden_bias:
  if args.convolution_type == "full":
    raise "Unsupported: multiple hidden bias needs borderconv"
  rbm.W = borderconvparameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], 'W', name='W', shape_info=shape_info, var_fixed_border=rbm.v_border, shared_hidden_dims=1, alternative_gradient=True)
elif args.convolution_type == "full":
  rbm.W = borderconvparameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], 'W', name='W', shape_info=shape_info, var_fixed_border=rbm.v_border, alternative_gradient=True)
elif args.convolution_type == "no":
  rbm.W = parameters.AdvancedProdParameters(rbm, [rbm.v, rbm.h], [3,1], 'W', name='W')
elif args.convolution_type == "fullnoborder":
  rbm.W = borderconvparameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], 'W', name='W', shape_info=shape_info)

# one bias per map (so shared across width and height):
if args.previous_layer_binary:
  rbm.bv = parameters.SharedBiasParameters(rbm, rbm.v, 3, 2, 'bv', name='bv')
else:
  rbm.bv = parameters.SharedQuadraticBiasParameters(rbm, rbm.v, 3, 2, 'bv', name='bv')

if args.non_shared_bias:
  rbm.bh = parameters.AdvancedBiasParameters(rbm, rbm.h, 3, 'bh', name='bh')
elif args.fixed_hidden_bias:
  rbm.bh = parameters.FixedBiasParameters(rbm, rbm.h, value=-10)
  rbm.bh.var = theano.shared(value=initial_bh, name='bh')
elif args.convolution_type == "no":
  rbm.bh = parameters.BiasParameters(rbm, rbm.h, 'bh', name='bh')
elif args.multiple_hidden_bias:
  rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 4, 2, 'bh', name='bh')
else:
  rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 3, 2, 'bh', name='bh')

# labels
if args.ignore_labels:
  pass
elif args.convolution_type == "no":
  rbm.U = parameters.ProdParameters(rbm, [rbm.y, rbm.h], 'U', name='U')
elif args.multiple_hidden_bias:
  if args.no_U_pooling:
    rbm.U = parameters.AdvancedProdParameters(rbm, [rbm.y, rbm.h], [1, 4], 'U', name='U')
  else:
    rbm.U = parameters.SharedProdParameters(rbm, [rbm.y, rbm.h], 4, 2, 'U', name='U', pooling_operator=pooling_operator)
else:
  if args.no_U_pooling:
    rbm.U = parameters.AdvancedProdParameters(rbm, [rbm.y, rbm.h], [1, 3], 'U', name='U')
  else:
    rbm.U = parameters.SharedProdParameters(rbm, [rbm.y, rbm.h], 3, 2, 'U', name='U', pooling_operator=pooling_operator)
if not args.ignore_labels:
  rbm.by = parameters.BiasParameters(rbm, rbm.y, 'by', name='by')

initial_vmap = { rbm.v: T.tensor4('v') }
if not args.ignore_labels:
  initial_vmap[rbm.y] = T.matrix('y')
if args.convolution_type == "full":
  initial_vmap[rbm.v_border] = T.tensor4('v border')

print rbm


if not args.ignore_labels:
  dlo = objectives.discriminative_learning_objective(rbm, \
      visible_units = [rbm.v], \
      hidden_units = [rbm.h], \
      label_units = [rbm.y], \
      vmap = initial_vmap,
      pmap = pmap)
else:
  dlo = theano.shared(value=np.cast['float32'](0)) * theano.shared(value=np.cast['float32'](0))

# learning rate decay
learning_rate_with_decay = theano.shared(value=np.cast[theano.config.floatX](learning_rate), name='learning_rate')

# try to calculate weight updates using CD-1 stats
print ">> Constructing contrastive divergence updaters..."
k_train = args.k_train
k_eval = args.k_eval
print "k_train=%d   k_eval=%d" % (k_train, k_eval)
if not args.ignore_labels:
# s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v, rbm.y], hidden_units=[rbm.h], context_units=context_units, \
#                    k=k_train, mean_field_for_stats=[rbm.v, rbm.y], mean_field_for_gibbs=[rbm.v, rbm.y, rbm.h])
  s = stats.cd_stats(rbm, initial_vmap, pmap, visible_units=[rbm.v, rbm.y], hidden_units=[rbm.h], context_units=context_units, \
                     k=k_train, mean_field_for_stats=[rbm.v, rbm.y], mean_field_for_gibbs=[rbm.v, rbm.y])
else:
# s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], context_units=context_units, \
#                    k=k_train, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v, rbm.h])
  s = stats.cd_stats(rbm, initial_vmap, pmap, visible_units=[rbm.v], hidden_units=[rbm.h], context_units=context_units, \
                     k=k_train, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])
print "mean_field_for_gibbs without rbm.h !"

print "Stats eval"
if not args.ignore_labels:
  s_eval = stats.cd_stats(rbm, initial_vmap, pmap, visible_units=[rbm.v, rbm.y], hidden_units=[rbm.h], context_units=context_units, k=k_eval) # , mean_field_for_gibbs=[rbm.v], mean_field_for_stats=[rbm.v, rbm.h])
else:
  s_eval = stats.cd_stats(rbm, initial_vmap, pmap, visible_units=[rbm.v], hidden_units=[rbm.h], context_units=context_units, k=k_eval) # , mean_field_for_gibbs=[rbm.v], mean_field_for_stats=[rbm.v, rbm.h])

decayed_beta = theano.shared(value=np.cast['float32'](args.beta), name='beta')

def sparsity_penalty(rbm, hidden_units, v0_vmap, pmap, target):
  """
  Customised to support convolutional RBM's multiple hidden maps.
  """
  # complete units lists
  hidden_units = rbm.complete_units_list(hidden_units)

  # complete the supplied vmap
  v0_vmap = rbm.complete_vmap(v0_vmap)

  hidden_vmap = rbm.mean_field(hidden_units, v0_vmap, pmap)

  if len(hidden_units) > 1:
    raise "More than one set of hidden units."

  penalty_terms = []
  for hu in hidden_units:
    mean_activation = T.mean(hidden_vmap[hu], [0,2,3]) # mean over minibatch,x,y
#   penalty_terms.append(T.sum((mean_activation - target) ** 2))
    penalty_terms.append(target - mean_activation)

  total_penalty = penalty_terms[0] # sum(penalty_terms)
  return total_penalty

class LeeSparsityUpdater(morb.base.Updater):
  def get_update(self):
    return theano.printing.Print("sparsity update")(sparsity_penalty(rbm, [rbm.h], initial_vmap, pmap, args.target_sparsity))

if args.sparsity_lambda and args.target_sparsity:
  calc_sparsity_penalty = sparsity_penalty(rbm, [rbm.h], initial_vmap, pmap, args.target_sparsity)
# dlo = calc_sparsity_penalty

umap = {}
for var in rbm.variables:
  learning_rate_for_var = learning_rate_with_decay
  if args.learning_rate_bias and (var == rbm.bh.var or var == rbm.bv.var):
    learning_rate_for_var = args.learning_rate_bias

  pu = pmap[var]
  if args.ignore_labels:
    pu += learning_rate_for_var * (decayed_beta * updaters.CDUpdater(rbm, var, s))
  elif args.train_label_only_discriminative and var in (rbm.U.var, rbm.by.var):
    pu += args.train_label_only_discriminative * updaters.GradientUpdater(dlo, var, pmap=pmap)
  else:
    pu += learning_rate_for_var * (decayed_beta * updaters.CDUpdater(rbm, var, s) + (1 - decayed_beta) * updaters.GradientUpdater(dlo, var, pmap=pmap))
  if args.sparsity_lambda and args.target_sparsity and var == rbm.bh.var:
    print "using sparsity penalty!"
    # pu += args.sparsity_lambda * updaters.SelfUpdater(calc_sparsity_penalty) # , var, pmap=pmap)
    pu += args.sparsity_lambda * LeeSparsityUpdater(var)
  if args.bias_decay and var == rbm.bh.var:
    pu -= learning_rate_for_var * args.bias_decay * mb_size

  umap[pmap[var]] = pu

if args.goh_sparsity_lambda and args.goh_target_sparsity:
  sparsity_targets = { rbm.h: args.goh_target_sparsity }
  umap[rbm.W.var] = umap[rbm.W.var] + learning_rate * args.goh_sparsity_lambda * updaters.SparsityUpdater(rbm, rbm.W.var, sparsity_targets, s)
  umap[rbm.bh.var] = umap[rbm.bh.var] + learning_rate * args.goh_sparsity_lambda * updaters.SparsityUpdater(rbm, rbm.bh.var, sparsity_targets, s)

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
mse_v = monitors.reconstruction_mse(s, rbm.v)
if not args.ignore_labels:
  mse_y = monitors.reconstruction_mse(s, rbm.y)
else:
  mse_y = theano.shared(value=np.cast['float32'](0)) * theano.shared(value=np.cast['float32'](0))
m_data = s['data'][rbm.v]
m_model = s['model'][rbm.v]
e_data = rbm.energy(s['data'], pmap).mean()
e_model = rbm.energy(s['model'], pmap).mean()
h_data = s['data'][rbm.h]
h_model = s['model'][rbm.h]
if not args.ignore_labels:
  y_data = s['data'][rbm.y]
  y_model = s['model'][rbm.y]
else:
  y_data = theano.shared(value=np.cast['float32'](0)) * theano.shared(value=np.cast['float32'](0))
  y_model = theano.shared(value=np.cast['float32'](0)) * theano.shared(value=np.cast['float32'](0))
v_data = s['data'][rbm.v]
v_model = s['model'][rbm.v]

train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[mse_v, mse_y, e_data, e_model, h_data, h_model, v_model, dlo, v_data], name='train', mode=mode)
evalt = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[mse_v, mse_y, e_data, e_model, h_data, h_model, v_model, dlo, v_data], name='eval', train=False, mode=mode)

print ">> Compiling prediction..."
if not args.ignore_labels:
  predict = prediction.label_prediction(rbm, initial_vmap, pmap, \
      visible_units = [rbm.v], \
      label_units = [rbm.y], \
      hidden_units = [rbm.h],
      context_units = context_units,
      mb_size=mb_size, mode=mode,
      only_activation = True)

print ">> Compiling evaluation..."
umap_e = {}
for var in rbm.variables:
    umap_e[pmap[var]] = pmap[var] + learning_rate_with_decay * updaters.CDUpdater(rbm, var, s_eval)
t_eval = trainers.MinibatchTrainer(rbm, umap_e)
msev_eval = monitors.reconstruction_mse(s_eval, rbm.v)
if not args.ignore_labels:
  msey_eval = monitors.reconstruction_mse(s_eval, rbm.y)
else:
  msey_eval = theano.shared(value=np.cast['float32'](0)) * theano.shared(value=np.cast['float32'](0))
evaluate = t_eval.compile_function(initial_vmap, mb_size=mb_size, monitors=[msev_eval, msey_eval, s_eval['data'][rbm.v], s_eval['model'][rbm.v], s_eval['data'][rbm.h], s_eval['model'][rbm.h]], name='evaluate', train=False, mode=mode)


############################################
# TRAINING
############################################

print ">> Training for %d epochs..." % epochs
print

stats = {}

def record_stat(name, epoch, value):
    stats.setdefault(name, []).append((epoch, float(value)))


def confmat(x_set, y_set, x_set_border = None):
    expected = np.argmax(y_set, axis=1)
    if x_set_border is None:
      predicted = np.concatenate([np.argmax(y, axis=1) for y, in predict({ rbm.v: x_set })])
    else:
      predicted = np.concatenate([np.argmax(y, axis=1) for y, in predict({ rbm.v: x_set, rbm.v_border: x_set_border })])
    return Confmat(expected, predicted)
def record_stat(name, epoch, value):
    stats.setdefault(name, []).append((epoch, float(value)))


############################################
# RESUMING
############################################

first_epoch = 0

if args.resume_run:
  resume_run_statsfile = args.resume_run
  resume_run_id = re.search(r'stats-([0-9-]+)\.json', resume_run_statsfile).groups()[0]

  print ">> Resuming previous run %s" % resume_run_id

  # load stats
  with open(resume_run_statsfile) as f:
    stats = json.load(f)
  first_epoch = stats["msev_train"][-1][0]

  # load last weights
  print "   Resuming from epoch %d" % first_epoch
  with open("%s/rbm-%s-epoch-%d.pkl" % (os.path.dirname(resume_run_statsfile), resume_run_id, first_epoch)) as f:
    previous_vars = cPickle.load(f)

  # collect list of current parameters
  current_params = dict([ (param.name, param.var) for param in rbm.params_list if hasattr(param, "var") ])

  # restore parameter values from pkl
  for key, value in previous_vars.iteritems():
    if not key in current_params:
      raise "   %s set in pkl, but not in current rbm!" % key
    print "   - Restoring %s from pkl" % key
    current_params[key].set_value(value)
    del current_params[key]

  if not len(current_params) == 0:
    raise "   %s in rbm, but not in pkl!" % current_params.keys()[0]

  print



start_time = time.time()

for epoch in range(first_epoch, epochs):
    print "epoch: %6d, time = %.2f s" % (epoch, time.time() - start_time)

    # set current learning rate
    if args.learning_rate_decay_start:
      decay_duration = args.learning_rate_decay_end - args.learning_rate_decay_start
      decay_progress = (epoch - args.learning_rate_decay_start) / float(decay_duration)
      decay_progress = max(0, min(1, decay_progress))
      print "decay progress", decay_progress
      learning_rate_with_decay.set_value(args.learning_rate * (1 - decay_progress) \
                                         + args.learning_rate_decay_end_rate * decay_progress)
      print "current learning rate: %f" % learning_rate_with_decay.get_value()

    # update beta
    decayed_beta.set_value(args.beta * (args.beta_decay ** epoch))
    print "current beta: %.2f" % decayed_beta.get_value()

    var_prev_vals = {}
    for var in rbm.variables:
      var_prev_vals[var] = pmap[var].get_value()

    ## TRAIN
    if not args.ignore_labels:
      if args.convolution_type == "full":
        costs = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in train({ rbm.v: train_set_x, rbm.y: train_set_y, rbm.v_border: train_set_x_border }, shuffle_batches_rng=numpy_rng)]
      else:
        costs = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in train({ rbm.v: train_set_x, rbm.y: train_set_y }, shuffle_batches_rng=numpy_rng)]
    else:
      if args.convolution_type == "full":
        costs = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in train({ rbm.v: train_set_x, rbm.v_border: train_set_x_border }, shuffle_batches_rng=numpy_rng)]
      else:
        costs = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in train({ rbm.v: train_set_x }, shuffle_batches_rng=numpy_rng)]

    var_diffs = {}
    for var in rbm.variables:
      var_diffs[var] = np.sum(np.abs(var_prev_vals[var] - pmap[var].get_value()))
      record_stat("update_"+str(var), epoch, var_diffs[var])
    print "epoch: %6d, updates since previous epoch: %s" % (epoch, repr(var_diffs))

    # average over minibatches
    msev_train = np.mean([ mv for mv,my,ed,em,hd,hm,vm,d,vd in costs ])
    msey_train = np.mean([ my for my,my,ed,em,hd,hm,vm,d,vd in costs ])
    edata_train = np.mean([ ed for mv,my,ed,em,hd,hm,vm,d,vd in costs ])
    emodel_train = np.mean([ em for m,my,ed,em,hd,hm,vm,d,vd in costs ])
    dlo_train = np.mean([ d for m,my,ed,em,hd,hm,vm,d,vd in costs ])

    print "epoch: %6d, train:  MSE v = %.6f, MSE y = %.6f, data energy = %.2f, model energy = %.2f, dlo = %.2f" % (epoch, msev_train, msey_train, edata_train, emodel_train, dlo_train)

    # record stats
    record_stat("msev_train", epoch, msev_train)
    record_stat("msey_train", epoch, msey_train)
    record_stat("edata_train", epoch, edata_train)
    record_stat("emodel_train", epoch, emodel_train)
    record_stat("dlo_train", epoch, dlo_train)

    if math.isnan(msev_train) or math.isinf(msev_train):
      print "NaN or inf"
      sys.exit()

    ## EVALUATE?
    if epoch % args.evaluate_every == 0:
      ## EVALUATE ON TEST SET
      if not args.ignore_labels:
        if args.convolution_type == "full":
          eval_res = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in evalt({ rbm.v: test_set_x, rbm.y: test_set_y, rbm.v_border: test_set_x_border })]
        else:
          eval_res = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in evalt({ rbm.v: test_set_x, rbm.y: test_set_y })]
      else:
        if args.convolution_type == "full":
          eval_res = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in evalt({ rbm.v: test_set_x, rbm.v_border: test_set_x_border })]
        else:
          eval_res = [(mv,my,ed,em,hd,hm,vm,d,vd) for mv,my,ed,em,hd,hm,vm,d,vd in evalt({ rbm.v: test_set_x })]

      # average over minibatches
      msev_test = np.mean([ mv for mv,my,ed,em,hd,hm,vm,d,vd in eval_res ])
      msey_test = np.mean([ my for mv,my,ed,em,hd,hm,vm,d,vd in eval_res ])
      edata_test = np.mean([ ed for mv,my,ed,em,hd,hm,vm,d,vd in eval_res ])
      emodel_test = np.mean([ em for mv,my,ed,em,hd,hm,vm,d,vd in eval_res ])
      dlo_test = np.mean([ d for m,my,ed,em,hd,hm,vm,d,vd in eval_res ])

      # record stats
      record_stat("msev_test", epoch, msev_test)
      record_stat("msey_test", epoch, msey_test)
      record_stat("edata_test", epoch, edata_test)
      record_stat("emodel_test", epoch, emodel_test)
      record_stat("dlo_test", epoch, dlo_test)

      print "epoch: %6d, test:   MSE v = %.6f, MSE y = %.6f, data energy = %.2f, model energy = %.2f, dlo = %.2f" % (epoch, msev_test, msey_test, edata_test, emodel_test, dlo_test)


    ## PLOT?
    if epoch % args.plot_every == 0:
      if args.multiple_hidden_bias:
        h_data = costs[0][4][0][0]
        h_model = costs[0][5][0][0]
      else:
        h_data = costs[0][4][0]
        h_model = costs[0][5][0]

      if not os.path.exists("img/%s" % run_id):
        os.mkdir("img/%s" % run_id)

      plt.figure(num=2)
      plt.clf()
      W_value = pmap[rbm.W.var].get_value()
      if args.convolution_type == "no":
        W_value = np.rollaxis(W_value, W_value.ndim-1, 0)
      plot_filters(W_value, normalise_filter=True)
      ddist = filter_dot_distance(W_value)
      print "filter dot distance: %f" % ddist
      plt.title("Filters (epoch=%d) d=%f" % (epoch, ddist))
      plt.savefig("img/%s/%s-filters-%d.png" % (run_id, run_id, epoch))

      print "norm(rbm.bv) = %f    norm(rbm.bh) = %f    norm(rbm.W) = %f" % (np.linalg.norm(pmap[rbm.bv.var].get_value()), np.linalg.norm(pmap[rbm.bh.var].get_value()), np.linalg.norm(W_value))

      if not args.convolution_type == "no":
        plt.figure(num=8)
        plt.clf()
        f, ax = plt.subplots(num=8, nrows=2, ncols=2)
        # print "h_data[0]", h_data[0]
        ax[0,0].imshow(h_data[0], cmap=plt.cm.gray, interpolation='nearest')
        ax[0,0].set_title("h_data 0")
        if h_data.shape[0] > 1: # hidden_maps > 1:
          ax[1,0].imshow(h_data[1], cmap=plt.cm.gray, interpolation='nearest')
          ax[1,0].set_title("h_data 1")
        if h_data.shape[0] > 2: # hidden_maps > 2:
          ax[0,1].imshow(h_data[2], cmap=plt.cm.gray, interpolation='nearest')
          ax[0,1].set_title("h_data 2")
        if h_data.shape[0] > 3: # hidden_maps > 3:
          ax[1,1].imshow(h_data[3], cmap=plt.cm.gray, interpolation='nearest')
          ax[1,1].set_title("h_data 3")
        plt.suptitle("@ %d" % (epoch))
        plt.savefig("img/%s/%s-filters-%d-featmaps.png" % (run_id, run_id, epoch))

      plt.draw()
      plt.show()

      print "  h_data norm=",        np.linalg.norm(h_data),
      print "  h_model norm=",       np.linalg.norm(h_model),
      print "  v_train_data norm=",  np.linalg.norm(train_set_x[0,0]),
      print "  v_train_model norm=", np.linalg.norm(costs[0][5])

      ## CLASSIFICATION
      if not args.ignore_labels:
        if args.convolution_type == "full":
          confmat_train = confmat(train_set_x, train_set_y, train_set_x_border)
          confmat_test = confmat(test_set_x, test_set_y, test_set_x_border)
        else:
          confmat_train = confmat(train_set_x, train_set_y)
          confmat_test = confmat(test_set_x, test_set_y)

        accuracy_train = confmat_train.accuracy()
        accuracy_test = confmat_test.accuracy()

        record_stat("accuracy_train", epoch, accuracy_train)
        record_stat("accuracy_test", epoch, accuracy_test)

        print "epoch: %6d, train accuracy = %.4f, test accuracy = %.4f" % (epoch, accuracy_train, accuracy_test)
        print "epoch: %6d, confusion matrix train:" % (epoch)
        print confmat_train
        print "epoch: %6d, confusion matrix test:" % (epoch)
        print confmat_test



    ## TEST?
    if epoch % args.test_every == 0:
      v_train_model = costs[0][6]
      v_train_data = costs[0][8]

      ## EVALUATE ON TEST SET
      if not args.ignore_labels:
        if args.convolution_type == "full":
          eval_res = [(mv, my, m_data, m_model, h_data, h_model) for mv, my, m_data, m_model, h_data, h_model in evaluate({ rbm.v: test_set_x, rbm.y: test_set_y, rbm.v_border: test_set_x_border })]
        else:
          eval_res = [(mv, my, m_data, m_model, h_data, h_model) for mv, my, m_data, m_model, h_data, h_model in evaluate({ rbm.v: test_set_x, rbm.y: test_set_y })]
      else:
        if args.convolution_type == "full":
          eval_res = [(mv, my, m_data, m_model, h_data, h_model) for mv, my, m_data, m_model, h_data, h_model in evaluate({ rbm.v: test_set_x, rbm.v_border: test_set_x_border })]
        else:
          eval_res = [(mv, my, m_data, m_model, h_data, h_model) for mv, my, m_data, m_model, h_data, h_model in evaluate({ rbm.v: test_set_x })]

      # PLOTTING
      plt.figure(11)
      plt.clf()
      plt.plot([e for e,v in stats["msev_train"]], [v for e,v in stats["msev_train"]], label='train')
      plt.plot([e for e,v in stats["msev_test"]], [v for e,v in stats["msev_test"]], label='test')
      plt.title("MSE v")
      plt.legend()
      plt.draw()
      plt.savefig("img/%s/%s-eval-%d-msev.png" % (run_id, run_id, epoch))
      
      plt.figure(2)
      plt.clf()
      plt.plot([e for e,v in stats["msey_train"]], [v for e,v in stats["msey_train"]], label='train')
      plt.plot([e for e,v in stats["msey_test"]], [v for e,v in stats["msey_test"]], label='test')
      plt.title("MSE y")
      plt.legend()
      plt.draw()
      plt.savefig("img/%s/%s-eval-%d-msey.png" % (run_id, run_id, epoch))

      plt.figure(11)
      plt.clf()
      plt.plot([e for e,v in stats["update_W"]], [v for e,v in stats["update_W"]], label='train')
      plt.title("change in W")
      plt.legend()
      plt.draw()
      plt.savefig("img/%s/%s-eval-%d-update-W.png" % (run_id, run_id, epoch))
      
      if not args.ignore_labels:
        plt.figure(3)
        plt.clf()
        plt.plot([e for e,v in stats["accuracy_train"]], [v for e,v in stats["accuracy_train"]], label='train')
        plt.plot([e for e,v in stats["accuracy_test"]], [v for e,v in stats["accuracy_test"]], label='test')
        plt.title("Classification accuracy")
        plt.legend()
        plt.draw()
        plt.savefig("img/%s/%s-eval-%d-accuracy.png" % (run_id, run_id, epoch))
      
      plt.figure(4)
      plt.clf()
      plt.plot([e for e,v in stats["edata_train"]], [v for e,v in stats["edata_train"]], label='train / data')
      plt.plot([e for e,v in stats["emodel_train"]], [v for e,v in stats["emodel_train"]], label='train / model')
      plt.plot([e for e,v in stats["edata_test"]], [v for e,v in stats["edata_test"]], label='test / data')
      plt.plot([e for e,v in stats["emodel_test"]], [v for e,v in stats["emodel_test"]], label='test / model')
      plt.title("energy")
      plt.legend()
      plt.draw()
      plt.savefig("img/%s/%s-eval-%d-energy.png" % (run_id, run_id, epoch))

      plt.figure(4)
      plt.clf()
      plt.plot([e for e,v in stats["dlo_train"]], [v for e,v in stats["dlo_train"]], label='train')
      plt.plot([e for e,v in stats["dlo_test"]], [v for e,v in stats["dlo_test"]], label='test')
      plt.title("discriminative objective")
      plt.legend()
      plt.draw()
      plt.savefig("img/%s/%s-eval-%d-dlo.png" % (run_id, run_id, epoch))

      # draw filters
      plt.figure(num=5)
      plt.clf()
      W_value = pmap[rbm.W.var].get_value()
      if args.convolution_type == "no":
        W_value = np.rollaxis(W_value, W_value.ndim-1, 0)
      plot_filters(W_value, normalise_filter=True)
      ddist = filter_dot_distance(W_value)
      print "filter dot distance: %f" % ddist
      plt.title("Filters (epoch=%d) d=%f" % (epoch, ddist))
      plt.savefig("img/%s/%s-filters-%d.png" % (run_id, run_id, epoch))

      plt.figure(num=1, figsize=(plt.rcParams['figure.figsize'][0], plt.rcParams['figure.figsize'][1]*3))
      plt.clf()
      plt.gcf().set_size_inches(plt.rcParams['figure.figsize'][0], plt.rcParams['figure.figsize'][1]*3)
      f, ax = plt.subplots(num=1, nrows=7, ncols=3)
      for i in xrange(0,min(v_train_data.shape[0], 7)):
        ax[i,0].imshow(v_train_data[i,0], cmap=plt.cm.gray, interpolation='nearest')
        ax[i,0].set_title("Training sample", fontsize=6)
        ax[i,1].imshow(v_train_model[i,0], cmap=plt.cm.gray, interpolation='nearest')
        ax[i,1].set_title("Reconstruction\n(k=1, mean field)", fontsize=6)
        ax[i,2].imshow(m_model[i,0], cmap=plt.cm.gray, interpolation='nearest')
        ax[i,2].set_title("Reconstruction\n(k=%d, sampled)" % k_eval, fontsize=6)
      plt.suptitle("@ %d" % (epoch))
      plt.savefig("img/%s/%s-eval-%d-samples.png" % (run_id, run_id, epoch))


      ## SAVE
      with open("results/stats-%s.json" % run_id, "w") as f:
        f.write(json.dumps(stats))
      with open("results/rbm-%s-epoch-%d.pkl" % (run_id, epoch), "w") as f:
        d = dict([ (param.name, pmap[param.var].get_value()) for param in rbm.params_list if hasattr(param, "var") ])
        if args.convolution_type == "no":
          d['W'] = np.rollaxis(d['W'], d['W'].ndim-1, 0)
        cPickle.dump(d, f)

    print "epoch: %6d, time = %.2f s" % (epoch, time.time() - start_time)

    # garbage collection
    gc.collect()

    if os.path.exists("STOP"):
      break

