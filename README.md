Convolutional classification restricted Boltzmann machines
==========================================================
These scripts have been used to train and evaluate convolutional
classification restricted Boltzmann machines.

The code provided here was used for the experiments in

> Combining Generative and Discriminative Representation Learning  
> for Lung CT Analysis with Convolutional Restricted Boltzmann Machines  
> by Gijs van Tulder and Marleen de Bruijne  
> in IEEE Transactions on Medical Imaging (2016)  
> http://dx.doi.org/10.1109/TMI.2016.2526687  

If you have any questions, comments or are using this code for something
interesting, we would love to know. We would also appreciate it if you
cite our paper if you use this code for your own publications.

For the most recent version of these files, see

> http://vantulder.net/code/2016/tmi-ccrbm/

February 2016

Gijs van Tulder  
http://vantulder.net/

Biomedical Imaging Group Rotterdam  
Erasmus MC, Rotterdam, the Netherlands  
http://www.bigr.nl/



Data
----
The experiments were done on data from a public dataset for interstital
lung diseases, which is described in

> Building a reference multimedia database for interstitial lung diseases  
> by Adrien Depeursinge et al.  
> in Computerized Medical Imaging and Graphics (April 2012)  
> http://dx.doi.org/10.1016/j.compmedimag.2011.07.003

Once you have a copy of this dataset, you can use the MATLAB scripts in
`patch-preprocessing/` to extract patches.


Requirements
------------
We have used this code with:

  * Python 2.7 with NumPy, SciPy, scikit-learn, matplotlib
  * Theano 0.7

The code uses a modified version of Morb, a modular RBM implementation
in Theano. The modifications include support for classification RBMs.

You will need Ruby to run the glue scripts that generate the parameter
sets and schedule the experiments.

See `morb-repo/` for details, or go to https://github.com/gvtulder/morb

The original version of Morb by Sander Dieleman can be found at
https://github.com/benanne/morb


Components
----------
The main components are:

  * `exp_train_rbm.py`: trains an RBM
  * `exp_save_features.py`: loads an RBM and extracts and saves features
  * `exp_rbm_classification.py`: loads an RBM and performs classification
  * `experiment_random_forest.py`: loads a dataset and trains a random forest

These scripts are fairly generic and take a set of parameters to run. To run
the actual experiments, we used two other scripts:

  * `generate-recipes.rb`: creates the commands to train RBMs with various
    parameters and training and test folds
  * `experiment-planner.rb`: runs the random forest evaluations after the
    RBMs have been trained

As baselines, we used Leung-Malik and Schmidt filter banks. Code to generate
these filters can be found in the `filter-banks/` directory.


Copyright and license
---------------------
Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands.  
This code is licensed under the MIT license. See LICENSE for details.

