__author__ = "Dimitri Bouche - dimi.bouche@gmail.com"


import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os
import numpy as np

import hmm
import em as em_dm3

from Code_DM2 import em, kmeans, utils

importlib.reload(hmm)
importlib.reload(em_dm3)


# Plotting parameters
plt.rcParams.update({'font.size': 25})

# Reload module (for developpement)

# Load the data
path = os.getcwd() + "/data/"
data_train = pd.read_table(path + "EMGaussian.data", header=None, sep=" ")
data_test = pd.read_table(path + "EMGaussian.test", header=None, sep=" ")
x = data_train.values.T
xtest = data_test.values.T
xall = np.concatenate((x, xtest), axis=1)


# Run EM with general covariance matrices
# Initialization with kmeans
mus_0, z = kmeans.iterate_kmeans(x, 4)
pi_0 = utils.cluster_repartition(z)
sigmas_0 = utils.clusters_cov(x, z)
pi, mus, sigmas, qs = em.iterate_em(x, pi_0, mus_0, sigmas_0, 200, 0.00001, diag=False)
# Predict labels using the parameters learned by EM
ztrain = em.assign_cluster(x, pi, mus, sigmas)
# Plot the results
fig3, axes3 = plt.subplots(1, 2)
utils.plot_clusters_ellipses(x, mus, sigmas, 1.5, ztrain, axes3[0])
# Predict labels using the parameters learned by EM on all data (train + test)
zall = em.assign_cluster(xall, pi, mus, sigmas)
# Plot the results
utils.plot_clusters(xall, mus, zall, axes3[1])
axes3[0].set_title("EM general - Train data")
axes3[1].set_title("EM general - All data")



A = 0.25 * np.ones((4, 4))
pi0 = 0.25 * np.ones(4)
test_alpha = hmm.alpha_recursion(x, A, pi0, mus, sigmas)

test_beta = hmm.beta_recursion(x, A, mus, sigmas)

sm = hmm.log_smoothing_delta(test_alpha, test_beta, 50)

log_xi = hmm.log_smoothing_xi(x, A, test_alpha, test_beta, 50, mus, sigmas)

xi_tensor = hmm.smoothing_xi_tensor(x, A, test_alpha, test_beta, mus, sigmas)

delta_mat = hmm.smoothing_delta_mat(test_alpha, test_beta)

pitest = em_dm3.pi_update(delta_mat)

mustest = em_dm3.mus_update(x, delta_mat)

Atest = em_dm3.A_update(xi_tensor)

sigtest = em_dm3.sigmas_update(x, delta_mat, mustest)

test_em = em_dm3.iterate_em(x, pi0, A, mus, sigmas, 1000, 0.001)