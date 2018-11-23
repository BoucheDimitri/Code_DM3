import numpy as np

import hmm


def pi_update(post_probas_mat):
    pi_kplus1 = post_probas_mat[:, 0]
    pi_kplus1 *= (1 / np.sum(pi_kplus1))
    return pi_kplus1


def mus_update(umat, post_probas_mat):
    K = post_probas_mat.shape[0]
    d = umat.shape[0]
    mus_kplus1 = np.zeros((d, K))
    for i in range(0, K):
        mus_kplus1[:, i] = np.sum(post_probas_mat[i, :] * umat, axis=1)
        mus_kplus1[:, i] *= 1 / (np.sum(post_probas_mat[i, :]))
    return mus_kplus1


def A_update(post_trans_tensor):
    A_kplus1 = np.sum(post_trans_tensor, axis=2)
    K = A_kplus1.shape[0]
    for i in range(0, K):
        A_kplus1[i, :] *= (1 / np.sum(A_kplus1[i, :]))
    return A_kplus1


def sigmas_update(umat, post_probas_mat, mus_kplus1):
    d = mus_kplus1.shape[0]
    T = umat.shape[1]
    K = post_probas_mat.shape[0]
    sigmas_kplus1 = []
    for i in range(0, K):
        matsum_tensor = np.zeros((T, d, d))
        for t in range(0, T):
            center_vec = umat[:, t].reshape((d, 1)) - mus_kplus1[:, i].reshape((d, 1))
            matsum_tensor[t, :, :] = np.dot(center_vec, center_vec.T)
        sigmas_kplus1.append(((post_probas_mat[i, :] * matsum_tensor).sum(axis=0)) / np.sum(post_probas_mat[i, :]))
    return sigmas_kplus1


def pz_given_x(x, pi, mus, sigmas):
    """
    p(z|x, pi, mus, sigmas)

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi (np.ndarray): multinomial mixture distribution, (ngaussians, )
        mus (np.ndarray): the mus stacked in columns (nfeatures, ngaussians)
        sigmas (list): list of covariance matrices, len(sigmas) = ngaussians and sigmas[j] = sigma_j

    Returns:
        np.ndarray: matrix which ij-th entry is p(z=j|x_i, pi, mus, sigmas)
    """
    k = pi.shape[0]
    n = x.shape[1]
    pzgx = np.zeros((k, n))
    for j in range(k):
        for i in range(n):
            pzgx[j, i] = pi[j] * multi_gaussian(x[:, i], mus[:, j], sigmas[j])
    for i in range(0, n):
        pzgx[:, i] *= (1 / np.sum(pzgx[:, i]))
    pzgx /= pzgx.sum(0)
    return pzgx


def log_gmatrix(x, mus, sigmas):
    """
    Matrix which entries are the log p(x_i|z=j, mus, sigmas)

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        mus (np.ndarray): the mus stacked in columns (nfeatures, ngaussians)
        sigmas (list): list of covariance matrices, len(sigmas) = ngaussians and sigmas[j] = sigma_j

    Returns:
        np.ndarray: Matrix which entries are the log p(x_i|z=j, mus, sigmas)
    """
    n = x.shape[1]
    k = len(sigmas)
    gmat = np.zeros((k, n))
    for i in range(0, n):
        for j in range(0, k):
            gmat[j, i] = np.log(multi_gaussian(x[:, i], mus[:, j], sigmas[j]))
    return gmat


def e_computation(x, pi_t, mus_t, sigmas_t, pi_tplus1, mus_tplus1, sigmas_tplus1):
    """
    Computation of E_(z | x, mus_tplus1, sigmas_tplus1) [log p(x, z|pi_t, mus_t, sigmas_t)]

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi_t (np.ndarray): multinomial mixture distribution at t (ngaussians, )
        mus_t (np.ndarray): the mus at t stacked in columns (nfeatures, ngaussians)
        sigmas_t (list): list of covariance matrices at t, len(sigmas) = ngaussians
        pi_tplus1 (np.ndarray): multinomial mixture distribution at t + 1 (ngaussians, )
        mus_tplus1 (np.ndarray): the mus at t + 1 stacked in columns (nfeatures, ngaussians)
        sigmas_tplus1 (list): list of covariance matrices at t + 1, len(sigmas) = ngaussians

    Returns:
        float: E_(z | x, mus_tplus1, sigmas_tplus1) [log p(x, z|pi_t, mus_t, sigmas_t)]
    """
    pzgx = pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_term = np.dot(np.sum(pzgx, axis=1), np.log(pi_tplus1))
    gmat = log_gmatrix(x, mus_tplus1, sigmas_tplus1)
    mus_sigs_term = np.sum(gmat * pzgx)
    return pi_term + mus_sigs_term


def m_step_pi(pzgx):
    """
    M update for pi

    Params:
        pzgx (np.ndarray): matrix which ij-th entry is p(z=j|x_i, pi, mus, sigmas)

    Returns:
        np.ndarray: the updated pi vector
    """
    pi_tplus1 = (1 / np.sum(pzgx)) * np.sum(pzgx, axis=1)
    return pi_tplus1


def m_step_mus(x, pzgx):
    """
    M update for mus

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pzgx (np.ndarray): matrix which ij-th entry is p(z=j|x_i, pi, mus, sigmas)

    Returns:
        np.ndarray: the updated mus stacked in columns
    """
    d = x.shape[0]
    k = pzgx.shape[0]
    mus_tplus1 = np.zeros((d, k))
    for j in range(0, k):
        mus_tplus1[:, j] = (1 / np.sum(pzgx[j, :])) * np.sum(pzgx[j, :] * x, axis=1)
    return mus_tplus1


def m_step_sigmas_diag(x, pzgx, mus_tplus1):
    """
    M update for sigmas in the case where they are proportionnal to identity

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pzgx (np.ndarray): matrix which ij-th entry is p(z=j|x_i, pi, mus, sigmas)
        mus_tplus1 (np.ndarray): already optimized mus stacked in columns (nfeatures, ngaussians)

    Returns:
        list: list of updated covariance matrices
    """
    n = x.shape[1]
    k = pzgx.shape[0]
    d = x.shape[0]
    sigmas1d = np.zeros((k, ))
    for j in range(0, k):
        for i in range(0, n):
            xcij = x[:, i] - mus_tplus1[:, j]
            sigmas1d[j] += pzgx[j, i] * np.dot(xcij.T, xcij)
        sigmas1d[j] *= (1 / np.sum(pzgx[j, :]))
    sigmas_tplus1 = []
    for j in range(0, k):
        sigmas_tplus1.append((1 / d) * sigmas1d[j] * np.eye(d))
    return sigmas_tplus1


def m_step_sigmas(x, pzgx, mus_tplus1):
    """
    M update for sigmas in the general covariance case

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pzgx (np.ndarray): matrix which ij-th entry is p(z=j|x_i, pi, mus, sigmas)
        mus_tplus1 (np.ndarray): already optimized mus stacked in columns (nfeatures, ngaussians)

    Returns:
        list: list of updated covariance matrices
    """
    n = x.shape[1]
    k = pzgx.shape[0]
    d = x.shape[0]
    sigmas_tplus1 = []
    for j in range(0, k):
        sigmas_tplus1.append(np.zeros((d, d)))
    for j in range(0, k):
        for i in range(0, n):
            xcij = (x[:, i] - mus_tplus1[:, j]).reshape(d, 1)
            sigmas_tplus1[j] += pzgx[j, i] * np.dot(xcij, xcij.T)
        sigmas_tplus1[j] *= (1 / np.sum(pzgx[j, :]))
    return sigmas_tplus1


def m_step_diag(x, pi_t, mus_t, sigmas_t):
    """
    M step for all parameters in the proportionnal to identity case

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi_t (np.ndarray): multinomial mixture distribution at t (ngaussians, )
        mus_t (np.ndarray): the mus at t stacked in columns (nfeatures, ngaussians)
        sigmas_t (list): list of covariance matrices at t, len(sigmas) = ngaussians

    Returns:
        tuple: the updated parameters
    """
    pzgx = pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_tplus1 = m_step_pi(pzgx)
    mus_tplus1 = m_step_mus(x, pzgx)
    sigmas_tplus1 = m_step_sigmas_diag(x, pzgx, mus_tplus1)
    return pi_tplus1, mus_tplus1, sigmas_tplus1


def m_step(x, pi_t, mus_t, sigmas_t):
    """
    M step for all parameters in the general case

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi_t (np.ndarray): multinomial mixture distribution at t (ngaussians, )
        mus_t (np.ndarray): the mus at t stacked in columns (nfeatures, ngaussians)
        sigmas_t (list): list of covariance matrices at t, len(sigmas) = ngaussians

    Returns:
        tuple: the updated parameters
    """
    pzgx = pz_given_x(x, pi_t, mus_t, sigmas_t)
    pi_tplus1 = m_step_pi(pzgx)
    mus_tplus1 = m_step_mus(x, pzgx)
    sigmas_tplus1 = m_step_sigmas(x, pzgx, mus_tplus1)
    return pi_tplus1, mus_tplus1, sigmas_tplus1


def iterate_em(x, pi_0, mus_0, sigmas_0, maxit, epsilon, diag=False):
    """
    EM algorithm iteration

    Params:
        x (np.ndarray): datamatrix (nfeatures, nsamples)
        pi_0 (np.ndarray): initial value for pi
        mus_0 (np.ndarray): initial mus
        sigmas_0 (list): initial covariance matrices
        maxit (int): maximum number of iterations
        epsilon (float): stopping criterion. Stops when the absolute value of the variation of e_computation(.) is inferior to epsilon
        diag (bool): covariance matrices proportionnal to identity or general case

    Returns:
        tuple: the updated parameters plus the history of the values of e_computation(.) during exec
    """
    qexpecs = [np.inf]
    pi_t, mus_t, sigmas_t = pi_0, mus_0, sigmas_0
    for t in range (0, maxit):
        if diag:
            pi_tplus1, mus_tplus1, sigmas_tplus1 = m_step_diag(x, pi_t, mus_t, sigmas_t)
        else:
            pi_tplus1, mus_tplus1, sigmas_tplus1 = m_step(x, pi_t, mus_t, sigmas_t)
        qexpec = e_computation(x, pi_t, mus_t, sigmas_t, pi_tplus1, mus_tplus1, sigmas_tplus1)
        qexpecs.append(qexpec)
        if np.abs(qexpecs[t + 1] - qexpecs[t]) < epsilon:
            return pi_tplus1, mus_tplus1, sigmas_tplus1, qexpecs[1:]
        pi_t, mus_t, sigmas_t = pi_tplus1, mus_tplus1, sigmas_tplus1
        print(t)
    return pi_tplus1, mus_tplus1, sigmas_tplus1, qexpecs[1:]