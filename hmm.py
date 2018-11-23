import numpy as np
import scipy.stats as stats


def log_plus(x):
    b = np.max(x)
    return b + np.log(np.sum(np.exp(x - b)))


def conditional_log_densities(u, mus, sigmas):
    k = mus.shape[1]
    densities = np.array([stats.multivariate_normal.logpdf(u, mus[:, i], sigmas[i]) for i in range(0, k)])
    return densities


# def conditional_densities(u, mus, sigmas):
#     k = mus.shape[1]
#     densities = np.array([stats.multivariate_normal.pdf(u, mus[:, i], sigmas[i]) for i in range(0, k)])
#     return densities


# def alpha_recursion_step_bis(u, A, scaled_alpha_t, mus, sigmas):
#     densities = conditional_densities(u, mus, sigmas)
#     alpha_tplus1 = densities * (np.dot(A.T, scaled_alpha_t))
#     c_tplus1 = (1 / np.sum(alpha_tplus1))
#     return c_tplus1 * alpha_tplus1, c_tplus1
#
#
# def alpha_recursion_bis(umat, A, pi0, mus, sigmas):
#     k = A.shape[0]
#     T = umat.shape[1]
#     scaled_alphas = np.zeros((k, T))
#     scaling_coefs = np.zeros(T)
#     fs = conditional_densities(umat[:, 0], mus, sigmas)
#     alpha0 = fs * np.log(pi0)
#     c0 = (1 / np.sum(alpha0))
#     scaled_alphas[:, 0] = fs * np.log(pi0)
#     scaling_coefs[0] = c0
#     for t in range(1, T):
#         scaled_alphas[:, t], scaling_coefs[t] = alpha_recursion_step_bis(umat[:, t], A, scaled_alphas[:, t - 1], mus, sigmas)
#     return (1 / np.cumprod(scaling_coefs)) * scaled_alphas
#

def alpha_recursion_step(u, A, log_alpha_t, mus, sigmas):
    k = log_alpha_t.shape[0]
    log_A = np.log(A)
    log_alpha_tplus1 = np.zeros(k)
    for i in range(0, k):
        log_prod = log_alpha_t + log_A[:, i]
        log_alpha_tplus1[i] = log_plus(log_prod)
    log_fs = conditional_log_densities(u, mus, sigmas)
    return log_fs + log_alpha_tplus1


def alpha_recursion(umat, A, pi0, mus, sigmas):
    k = A.shape[0]
    T = umat.shape[1]
    log_alphas = np.zeros((k, T))
    log_fs = conditional_log_densities(umat[:, 0], mus, sigmas)
    log_alphas[:, 0] = np.log(pi0) + log_fs
    for t in range(1, T):
        log_alphas[:, t] = alpha_recursion_step(umat[:, t], A, log_alphas[:, t - 1], mus, sigmas)
    return log_alphas


def beta_recursion_step(u, A, log_beta_tplus1, mus, sigmas):
    k = log_beta_tplus1.shape[0]
    log_A = np.log(A)
    log_beta_t = np.zeros(k)
    log_fs = conditional_log_densities(u, mus, sigmas)
    for i in range(0, k):
        log_prod = log_beta_tplus1 + log_A[i, :] + log_fs
        log_beta_t[i] = log_plus(log_prod)
    return log_beta_t


def beta_recursion(umat, A, mus, sigmas):
    k = A.shape[0]
    T = umat.shape[1]
    log_betas = np.zeros((k, T))
    for t in np.arange(T - 1, 0, -1):
        log_betas[:, t - 1] = beta_recursion_step(umat[:, t], A, log_betas[:, t], mus, sigmas)
    return log_betas


def log_smoothing_delta(log_alphas, log_betas, t):
    z = log_plus(log_alphas[:, t] + log_betas[:, t])
    return log_alphas[:, t] + log_betas[:, t] - z


def smoothing_delta_mat(log_alphas, log_betas):
    k = log_betas.shape[0]
    T = log_alphas.shape[1]
    log_delta_mat = np.zeros((k, T))
    for t in range(0, T):
        log_delta_mat[:, t] = log_smoothing_delta(log_alphas, log_betas, t)
    return np.exp(log_delta_mat)


def log_smoothing_xi(umat, A, log_alphas, log_betas, t, mus, sigmas):
    k = log_betas.shape[0]
    log_xi = np.zeros((k, k))
    log_A = np.log(A)
    log_fs = conditional_log_densities(umat[:, t + 1], mus, sigmas)
    for i in range(0, k):
        for j in range(0, k):
            log_xi[i, j] = log_A[i, j] + log_alphas[i, t] + log_betas[j, t + 1] + log_fs[j]
    z = log_plus(log_xi.flatten())
    return log_xi - z


def smoothing_xi_tensor(umat, A, log_alphas, log_betas, mus, sigmas):
    k = log_betas.shape[0]
    T = umat.shape[1]
    log_xi_tensor = np.zeros((k, k, T - 1))
    for t in range(0, T - 1):
        log_xi_tensor[:, :, t] = log_smoothing_xi(umat, A, log_alphas, log_betas, t, mus, sigmas)
    return np.exp(log_xi_tensor)


def omega_recursion_step(u, A, log_omega, mus, sigmas):
    K = log_omega.shape[0]
    log_fs = conditional_log_densities(u, mus, sigmas)
    max_term = np.max(log_omega.reshape((K, 1)) +  np.log(A), axis=0)
    return log_fs + max_term


def omega_recursion(umat, pi, A, mus, sigmas):
    T = umat.shape[1]
    K = pi.shape[0]
    log_omegas = np.zeros((K, T))
    log_fs = conditional_log_densities(umat[:, 0], mus, sigmas)
    log_omegas[:, 0] = np.log(pi) + log_fs
    for t in range(1, T):
        log_omegas[:, t] = omega_recursion_step(umat[:, t], A, log_omegas[:, t - 1], mus, sigmas)
    return log_omegas


def log_backtracking(umat, A, log_omegas):
    T = umat.shape[1]
    zs = np.zeros(T, dtype=int)
    zs[-1] = np.argmax(log_omegas[:, -1])
    for t in np.arange(T - 2, 0, -1):
        zs[t] = np.argmax(log_omegas[:, t] + np.log(A)[:, zs[t + 1]])
    return zs