__author__ = "Dimitri Bouche - dimi.bouche@gmail.com"


import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def random_init(xs, k):
    """
    Affect the data vectors to a random cluster

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        k (int): the number of clusters

    Returns:
        np.ndarray: random vector of assignement to clusters (nsamples, )
    """
    z = np.random.choice(np.array(range(0, k)), xs.shape[1])
    return z


def assign_xs(xs, mus):
    """
    Affect the data vectors to the cluster to which centroid they are the closest

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        mus (np.ndarray): the centroids (nfeatures, nclusters)

    Returns:
        np.ndarray: vector of assignement to clusters (nsamples, )
    """
    dists = distance.cdist(xs.T, mus.T)
    z = np.argmin(dists, axis=1)
    return z


def update_mus(xs, z):
    """
    Centroids update for kmeans

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        z (np.ndarray): vector of assignement to clusters (nsamples, )

    Returns:
        np.ndarray: vector of assignement to clusters (nsamples, )
    """
    k = np.unique(z).shape[0]
    mus = np.zeros((xs.shape[0], k))
    for i in range(0, k):
        inds = (z == i).astype(int)
        mus[:, i] = (1 / np.sum(inds)) * np.sum(xs * inds, axis=1)
    return mus


def iterate_kmeans(xs, k, nits=100, epsilon=0.001):
    """
    Iterate kmeans updates (1: centroid updates, 2: reassignement)

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        k (int): n clusters
        nits (int): numbers of iterations to perform
        epsilon (float): stopping criterion based on the delta of the distortion

    Returns:
        tuple: mus (centroid matrix), z (assignement vector)
    """
    z = random_init(xs, k)
    objs = [np.inf]
    for i in range(0, nits):
        mus = update_mus(xs, z)
        objs.append(distortion(xs, mus, z))
        z = assign_xs(xs, mus)
        if np.abs(objs[i+1] - objs[i]) < epsilon:
            return mus, z
        # print("Objective value: " + str(objs[i+1]))
    return mus, z


def distortion(xs, mus, z):
    """
    Global distortion measure

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        mus (np.ndarray): the centroids (nfeatures, nclusters)
        z (np.ndarray): vector of assignement to clusters (nsamples, )

    Returns:
        float: the global distortion measure
    """
    dists = distance.cdist(xs.T, mus.T) ** 2
    k = np.unique(z).shape[0]
    obj = 0
    for i in range(0, k):
        inds = (z == i).astype(int)
        obj += np.sum(dists[:, i] * inds)
    return obj


def compare_several_runs(xs, k, nsims, nits=100, epsilon=0.001):
    """
    Global distortion measure

    Params:
        xs (np.ndarray): the data matrix (nfeatures, nsamples)
        k (int): number of clusters
        nsims (int): number of runs of kmeans
        nits (int): numbers of iterations to perform
        epsilon (float): stopping criterion based on the delta of the distortion

    Returns:
        tuple: dictionnary of estimated centers and list of optimal distortion values reached
    """
    mus_dict = {}
    objs = []
    for i in range(0, k):
        mus_dict[i] = np.zeros((xs.shape[0], nsims))
    for j in range(0, nsims):
        mus, z = iterate_kmeans(xs, k, nits, epsilon)
        print(j)
        objs.append(distortion(xs, mus, z))
        for i in range(0, k):
            mus_dict[i][:, j] = mus[:, i]
    return mus_dict, objs


def plot_centroids(mus_dict, k):
    """
    Plot the centers dispertion from the dict of centers from the compare_several_runs function
    """
    for i in range(0, k):
        plt.scatter(mus_dict[i][0, :], mus_dict[i][1, :], c="k")
