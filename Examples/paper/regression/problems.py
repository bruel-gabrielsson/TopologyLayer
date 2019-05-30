# generate regression problems
import numpy as np


def generate_sinusoid(p, npeaks):
    t = np.linspace(0, 1, p)
    beta = np.cos((2*npeaks - 1) * np.pi * t)
    return beta


def generate_sawtooth(p, npeaks):
    t = np.linspace(0, npeaks, p)
    beta = np.mod(t, 1)*(npeaks-1) - 1.0
    return beta


def generate_boxcars(p, npeaks):
    t = np.linspace(0, npeaks, p)
    beta = 2.0*(np.mod(t, 1) > 0.5) - 1.0
    return beta


def generate_rips_problem(vs, p):
    """
    generate random vector sampled from vs with replacement
    """
    return np.random.choice(vs, p)


def generate_problem(beta, n, sigma):
    """
    generate regression problem
    """
    p = len(beta)
    X = np.random.randn(n,p)
    y = X.dot(beta) + sigma*np.random.randn(n)
    return X, y
