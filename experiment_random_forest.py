# Trains and evaluate a random forest classifier.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
import math
import os
import sys
import time
import cPickle

import numpy as np
import sklearn.datasets
import sklearn.ensemble
import sklearn.metrics
import sklearn.preprocessing
import sklearn.cross_validation

import helpers

np.set_printoptions(suppress=True)



import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("--experiment-id", metavar="ID", help="experiment ID", required=True)
parser.add_argument("--train-set", metavar="FILE", help="npz")
parser.add_argument("--train-patches", metavar="FILE", help="npy")
parser.add_argument("--train-labels", metavar="FILE", help="npy")
parser.add_argument("--validation-set", metavar="FILE", help="npz")
parser.add_argument("--validation-patches", metavar="FILE", help="npy")
parser.add_argument("--validation-labels", metavar="FILE", help="npy")
parser.add_argument("--test-set", metavar="FILE", help="npz")
parser.add_argument("--test-patches", metavar="FILE", help="npy")
parser.add_argument("--test-labels", metavar="FILE", help="npy")
parser.add_argument("--cross-validation", metavar="FOLDS", help="number of CV folds", type=int)
parser.add_argument("--n-estimators", metavar="N", help="number of trees")
parser.add_argument("--max-features", metavar="N", help="maximum number of features")
parser.add_argument("--max-depth", metavar="N", help="maximum depth", type=int)
parser.add_argument("--channel-mu-std-to-average", metavar="CHANNEL", type=int,
                    help="set mu, std for CHANNEL in eval to the average of train")
parser.add_argument("--seed", metavar="SEED", help="random seed", default="123")
parser.add_argument("--n-jobs", metavar="N", type=int, default=int(os.getenv("NSLOTS",1)))
parser.add_argument("--save-model", metavar="FILE")
parser.add_argument("--save-predictions", metavar="FILE")
args = parser.parse_args()
vargs = vars(args)


# load data
patches = {}
labels = {}
src = {}
for purpose in ("train", "validation", "test"):
  this_patches, this_labels, this_src = \
    helpers.load_data(vargs["%s_set"%purpose],
                      vargs["%s_patches"%purpose],
                      vargs["%s_labels"%purpose],
                      patches_dtype="float32",
                      labels_dtype="int32",
                      return_src=True)

  if len(this_patches) > 0:
    patches[purpose] = this_patches.reshape([this_patches.shape[0],-1])
    this_labels = this_labels.reshape([-1])
    if len(this_labels) > 0:
      labels[purpose] = this_labels
    if this_src is not None and len(this_src) > 0:
      src[purpose] = this_src

    print "%s patches:" % purpose, patches[purpose].shape
    if len(labels[purpose]) > 0:
      print "%s labels:" % purpose, labels[purpose].shape
    if this_src is not None and len(this_src) > 0:
      print "%s src:" % purpose, len(src[purpose])

patch_size = patches["train"].shape[1:]
for purpose in patches.keys():
  if patches[purpose].shape[1:] != patch_size:
    print "Different patch sizes in %s" % purpose
    sys.exit(1)

if len(labels) == 0:
  labels = None
  labels_oh = None
else:
  labels_oh = {}


if args.channel_mu_std_to_average is not None:
  # set intensity, mu and std for patches with a missing sequence to the average value
  average_intensity = np.mean(patches["train"][:, -12:-8], axis=0)
  average_mu = np.mean(patches["train"][:, -8:-4], axis=0)
  average_std = np.mean(patches["train"][:, -4:], axis=0)

  for purpose in ("validation", "test"):
    if purpose in patches:
      d = args.channel_mu_std_to_average
      patches[purpose][:, -12+d] = average_intensity[d]
      patches[purpose][:, -8+d] = average_mu[d]
      patches[purpose][:, -4+d] = average_std[d]


for n_estimators in args.n_estimators.split(","):
  n_estimators = int(n_estimators)
  for max_features in (args.max_features or "auto").split(","):
    if max_features != "auto":
      max_features = int(max_features)
    for seed in args.seed.split(","):
      seed = int(seed)

      print
      print "######"
      if args.max_depth is not None:
        print "n_estimators = %d   max_features = %s   seed = %d   max_depth = %d" % (n_estimators, str(max_features), seed, args.max_depth)
      else:
        print "n_estimators = %d   max_features = %s   seed = %d" % (n_estimators, str(max_features), seed)

      start_time = time.time()
      clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                    max_depth=args.max_depth,
                                                    oob_score=True, n_jobs=args.n_jobs, random_state=seed)
      print "Fitting model..."
      clf.fit(patches["train"], labels["train"])

      end_time = time.time()
      pretraining_time = (end_time - start_time)
      print ("Training RF took %f minutes" % (pretraining_time / 60.))
      print

      if args.save_predictions:
        predictions = {}

      print "Evaluating model..."
      if args.cross_validation is not None:
        cv = sklearn.cross_validation.KFold(n=len(patches["train"]), n_folds=args.cross_validation,
                                            shuffle=True, random_state=123)
        clf_for_cv = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                             max_depth=args.max_depth,
                                                             oob_score=True, n_jobs=args.n_jobs, random_state=seed)
        scores = sklearn.cross_validation.cross_val_score(clf_for_cv, patches["train"], labels["train"],
                                                          cv=cv, scoring="accuracy")
        print "crossval accuracy per fold:", scores
        print "crossval accuracy:", np.mean(scores)
        print
      for purpose in ("train", "validation", "test"):
        if purpose in patches:
          predict_labels = clf.predict(patches[purpose])
          print "%s accuracy:" % purpose, sklearn.metrics.accuracy_score(labels[purpose], predict_labels)
          print "%s confusion matrix:" % purpose
          print sklearn.metrics.confusion_matrix(labels[purpose], predict_labels)

          predict_proba = clf.predict_proba(patches[purpose])
          y_true = sklearn.preprocessing.label_binarize(labels[purpose], clf.classes_)
          if len(clf.classes_)==2:
            y_true = np.concatenate([y_true,1-y_true], axis=1)
#         if len(np.unique(y_true)) > 1:
#           roc_auc_score = sklearn.metrics.roc_auc_score(y_true, predict_proba, average=None)
#           print "%s roc_auc_score:" % purpose, roc_auc_score
          print

          if args.save_predictions:
            predictions["classes-%s" % purpose] = clf.classes_
            predictions["true-labels-%s" % purpose] = labels[purpose]
            predictions["predicted-labels-%s" % purpose] = predict_labels
            predictions["predicted-probability-%s" % purpose] = predict_proba
            if purpose in src:
              predictions["src-%s" % purpose] = src[purpose]

      print
      print

      if args.save_predictions:
        print "Writing predictions to %s" % args.save_predictions
        np.savez_compressed(args.save_predictions, **predictions)

      if args.save_model:
        print "Writing model to %s" % args.save_model
        from sklearn.externals import joblib
        joblib.dump(clf, args.save_model)


