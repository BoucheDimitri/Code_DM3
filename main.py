__author__ = "Dimitri Bouche - dimi.bouche@gmail.com"


import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os
import numpy as np

import hmm
import em as em_dm3

from Code_DM2 import em as em_dm2, kmeans, utils

# Reload module (for developpement)
importlib.reload(hmm)
importlib.reload(em_dm3)


# Plotting parameters
plt.rcParams.update({'font.size': 25})


# Load the data
path = os.getcwd() + "/data/"
data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
data_test = pd.read_table(path + "EMGaussian.test", header=None, sep=" ")
x = data_train.values.T
xtest = data_test.values.T
xall = np.concatenate((x, xtest), axis=1)


# Run EM with general covariance matrices (From DM2)
# Initialization with kmeans
mus_0, z = kmeans.iterate_kmeans(x, 4)
pi_0 = utils.cluster_repartition(z)
sigmas_0 = utils.clusters_cov(x, z)
pi, mus, sigmas, qs = em_dm2.iterate_em(x, pi_0, mus_0, sigmas_0, 200, 0.00001, diag=False)
# Predict labels using the parameters learned by EM
ztrain = em_dm2.assign_cluster(x, pi, mus, sigmas)
# Predict labels using the parameters learned by EM on all data (train + test)
zall = em_dm2.assign_cluster(xall, pi, mus, sigmas)
ztest = em_dm2.assign_cluster(xtest, pi, mus, sigmas)


# Initialization of matrix A
A0 = 0.25 * np.ones((4, 4))

# Initialization using the parameters learnt by em from DM2
pi0 = pi
mus0 = mus
sigmas0 = sigmas

# Perform EM for the HMM modelization
# We use the alpha and betas recursion to compute the necessary propabilities (see hmm.py file)
pi_hmm, A_hmm, mus_hmm, sigmas_hmm, qs = em_dm3.iterate_em(x, pi0, A0, mus0, sigmas0, maxit=1000, epsilon=0.001)

# Max product recursion (Viterbi algorithm) on training set
omegas_train = hmm.omega_recursion(x, pi_hmm, A_hmm, mus_hmm, sigmas_hmm)
# Backtracking step of the Viterbi algorithm for decoding
z_hmm_train = hmm.log_backtracking(x, A_hmm, omegas_train)
# Same thing on test set
omegas_test = hmm.omega_recursion(xtest, pi_hmm, A_hmm, mus_hmm, sigmas_hmm)
# Backtracking step of the Viterbi algorithm for decoding
z_hmm_test = hmm.log_backtracking(x, A_hmm, omegas_test)


# Compare EM for GMM and EM for HMM on training set
fig, axes = plt.subplots(1, 2)
utils.plot_clusters_ellipses(x, mus, sigmas, 1.5, ztrain, axes[0])
axes[0].set_title("EM for GMM - Training")
utils.plot_clusters_ellipses(x, mus_hmm, sigmas_hmm, 1.5, z_hmm_train, axes[1])
axes[1].set_title("EM for HMM - Training")

# Compare the fitted likelihoods
llk_train_gmm = em_dm2.log_likelihood(x, ztrain, pi, mus, sigmas)
llk_test_gmm = em_dm2.log_likelihood(xtest, ztest, pi, mus, sigmas)
llk_train_hmm = em_dm3.fitted_log_likelihood(x, z_hmm_train, pi_hmm, A_hmm, mus_hmm, sigmas_hmm)
llk_test_hmm = em_dm3.fitted_log_likelihood(xtest, z_hmm_test, pi_hmm, A_hmm, mus_hmm, sigmas_hmm)
print("Train GMM: " + str(llk_train_gmm))
print("Test GMM: " + str(llk_test_gmm))
print("Train HMM: " + str(llk_train_hmm))
print("Test HMM: " + str(llk_test_hmm))
print("Test/Train shift GMM: " + str((llk_test_gmm - llk_train_gmm) / (llk_train_gmm)))
print("Test/Train shift HMM: " + str((llk_test_hmm - llk_train_hmm) / (llk_train_hmm)))


# Compare EM for GMM and EM for HMM on training set
fig2, axes2 = plt.subplots(1, 2)
utils.plot_clusters_ellipses(xtest, mus, sigmas, 1.5, ztest, axes2[0])
axes2[0].set_title("EM for GMM - Test")
utils.plot_clusters_ellipses(xtest, mus_hmm, sigmas_hmm, 1.5, z_hmm_test, axes2[1])
axes2[1].set_title("EM for HMM - Test")