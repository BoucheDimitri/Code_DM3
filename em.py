import numpy as np

import hmm


def multi_gaussian(xvec, mu, sigma):
    """
    Pdf for multivariate Gaussian

    Params:
        xvec (np.ndarray): x
        mu (np.ndarray): mean vector
        sigma (np.ndarray): covariance matrix

    Returns:
        float: pdf at xvec
    """
    d = xvec.shape[0]
    norm = np.power(np.pi, 0.5 * d) * np.sqrt(np.linalg.det(sigma))
    sigma_inv = np.linalg.inv(sigma)
    intraexp = - 0.5 * np.dot(np.dot((xvec - mu).T, sigma_inv), xvec - mu)
    return (1 / norm) * np.exp(intraexp)


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
        sigmas_kplus1.append(((post_probas_mat[i, :].reshape((T, 1, 1)) * matsum_tensor).sum(axis=0)) / np.sum(post_probas_mat[i, :]))
    return sigmas_kplus1


def update_deltas_xis(umat, pi, A, mus, sigmas):
    log_alpha = hmm.alpha_recursion(umat, A, pi, mus, sigmas)
    log_beta = hmm.beta_recursion(umat, A, mus, sigmas)
    xi_tensor = hmm.smoothing_xi_tensor(umat, A, log_alpha, log_beta, mus, sigmas)
    delta_mat = hmm.smoothing_delta_mat(log_alpha, log_beta)
    return delta_mat, xi_tensor


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


def e_step(umat, delta_mat, xi_tensor, pi, A, mus, sigmas):
    pi_term = np.dot(delta_mat[:, 0], np.log(pi))
    xi_reduced = np.sum(xi_tensor, axis=2)
    A_term = np.sum(np.log(A) * xi_reduced)
    gmat = log_gmatrix(umat, mus, sigmas)
    fterm = np.sum(gmat * delta_mat)
    return pi_term + A_term + fterm


def m_step(umat, delta_mat, xi_tensor):
    new_mus = mus_update(umat, delta_mat)
    new_pi = pi_update(delta_mat)
    new_A = A_update(xi_tensor)
    new_sigmas = sigmas_update(umat, delta_mat, new_mus)
    return new_pi, new_A, new_mus, new_sigmas


def iterate_em(umat, pi_0, A_0, mus_0, sigmas_0, maxit, epsilon):
    """
    EM algorithm iteration

    Params:
        umat (np.ndarray): datamatrix (nfeatures, nsamples)
        pi_0 (np.ndarray): initial value for pi
        mus_0 (np.ndarray): initial mus
        sigmas_0 (list): initial covariance matrices
        maxit (int): maximum number of iterations
        epsilon (float): stopping criterion. Stops when the absolute value of the variation of e_computation(.) is inferior to epsilon

    Returns:
        tuple: the updated parameters plus the history of the values of e_computation(.) during exec
    """
    qexpecs = [np.inf]
    pi_k, A_k, mus_k, sigmas_k = pi_0, A_0, mus_0, sigmas_0
    for k in range(0, maxit):
        delta_mat, xi_tensor = update_deltas_xis(umat, pi_k, A_k, mus_k, sigmas_k)
        pi_kplus1, A_kplus1, mus_kplus1, sigmas_kplus1 = m_step(umat, delta_mat, xi_tensor)
        qexpec = e_step(umat, delta_mat, xi_tensor, pi_kplus1, A_kplus1, mus_kplus1, sigmas_kplus1)
        qexpecs.append(qexpec)
        if np.abs(qexpecs[k + 1] - qexpecs[k]) < epsilon:
            return pi_kplus1, A_kplus1, mus_kplus1, sigmas_kplus1, qexpecs[1:]
        pi_k, A_k, mus_k, sigmas_k = pi_kplus1, A_kplus1, mus_kplus1, sigmas_kplus1
        print(k)
    return pi_kplus1, A_kplus1, mus_kplus1, sigmas_kplus1, qexpecs[1:]
