import torch
import numpy as np
from problems import generate_problem


def penalized_ls(y, X, lam, penalty, verbose=False, lr=1e-1, maxiter=100):
    """
    minimize |y - X*beta|^2 + lam*penalty(beta)

    initializes beta with ls solution
    then runs maxiter iterations of GD
    """
    betals = np.linalg.lstsq(X, y, rcond=1e-4)[0]
    if penalty is None:
        return betals
    beta = torch.autograd.Variable(torch.tensor(betals, dtype=torch.float).view(-1,1), requires_grad=True)
    Xt = torch.tensor(X, dtype=torch.float)
    yt = torch.tensor(y, dtype=torch.float).view(-1,1)
    #print yt.shape

    l2 = torch.nn.MSELoss()

    optimizer = torch.optim.Adam([beta], lr=lr)
    for i in range(maxiter):
        optimizer.zero_grad()
        yhat = torch.matmul(Xt, beta)
        #print yhat.shape
        pen = penalty(beta)
        mse = l2(yt, yhat)
        loss = lam*pen + mse
        loss.backward()
        optimizer.step()
        if verbose and (i % 5 == 0):
            print ("[Iter %d] [penalty: %f] [mse : %f] [loss : %f]" % (i, pen.item(), mse.item(), loss.item()))

    return beta.detach().numpy().flatten()


def get_mse(beta0, beta1, n=10000):
    p = len(beta0)
    X = np.random.randn(n,p)
    y0 = X.dot(beta0)
    y1 = X.dot(beta1)
    return np.mean(np.power(y0 - y1,2))


def get_backward_error(beta0, beta1):
    return np.sum(np.power(beta0 - beta1, 2))/np.sum(np.power(beta0, 2))


def choose_best_lam(beta0, y, X, lams, pen):
    mses = []
    bes = []
    for lam in lams:
        beta1 = penalized_ls(y, X, lam, pen)
        mses.append(get_mse(beta0, beta1))
        bes.append(get_backward_error(beta0, beta1))
    ind = np.argmin(mses)
    return lams[ind], mses[ind], bes[ind]


def gen_mse_be(beta0, ns, lams, pen, sigma=0.05, ntrials=10):
    l = []
    mse = []
    be = []
    for n in ns:
        for trial in range(ntrials):
            X, y = generate_problem(beta0, n, 0.05)
            ln, msen, ben = choose_best_lam(beta0, y, X, lams, pen)
            l.append(ln)
            mse.append(msen)
            be.append(ben)
    l = np.array(l).reshape(-1,ntrials)
    mse = np.array(mse).reshape(-1,ntrials)
    be = np.array(be).reshape(-1,ntrials)
    return l, mse, be


def run_trials(beta0, n, sigma, lam, pen, ntrials=100, maxiter=100):
    """
    run ntrials regression problems

    return the mse of each trial.
    """
    # generate a regression problem y = X*beta0 + epsilon
    mses = []
    for i in range(ntrials):
        X, y = generate_problem(beta0, n, sigma)
        beta1 = penalized_ls(y, X, lam, pen, maxiter=maxiter)
        mses.append(np.linalg.norm(beta0 - beta1)**2)
    return np.array(mses)


def run_trials_ols(beta0, n, sigma, ntrials=100):
    """
    run ntrials regression problems

    return the mse of each trial.
    """
    mses = []
    for i in range(ntrials):
        X, y = generate_problem(beta0, n, sigma)
        beta1 = np.linalg.lstsq(X, y, rcond=1e-4)[0]
        mses.append(np.linalg.norm(beta0 - beta1)**2)
    return np.array(mses)


def get_stats(beta0, n, sigma, lam, pen, ntrials=100, maxiter=100):
    """
    run ntrials regression problems

    return mean of the mse, and 95% confidence interval
    """
    if pen is None:
        mses = run_trials_ols(beta0, n, sigma, ntrials=ntrials)
    else:
        mses = run_trials(beta0, n, sigma, lam, pen, ntrials=ntrials, maxiter=maxiter)
    mmean = np.mean(mses)
    qs = np.quantile(mses, [0.025, 0.875])
    return mmean, qs


def get_stats_lam(beta0, n, sigma, lams, pen, ntrials=100, maxiter=100, verbose=False):
    """
    for a set of regularization parameters lams, run trials to find mse
    and confidence intervals
    """
    if pen is None:
        return
    means = []
    qs = []
    for lam in lams:
        if verbose:
            print lam
        lmean, lqs = get_stats(beta0, n, sigma, lam, pen, ntrials=ntrials, maxiter=maxiter)
        means.append(lmean)
        qs.append(lqs)
    return np.array(means), np.array(qs)


def gen_snr_stats(beta0, n, sigmas, lams, pen, ntrials=100, maxiter=100):
    # first we determine the optimal value of lambda
    mses = []
    qs = []
    lamopt = []
    if pen is None:
        # just run ols
        for sigma in sigmas:
            smean, sqs = get_stats(beta0, n, sigma, None, None, ntrials=ntrials)
            mses.append(smean)
            qs.append(sqs)
    else:
        for sigma in sigmas:
            # get best value of lambda
            means, _ = get_stats_lam(beta0, n, sigma, lams, pen, ntrials=20, maxiter=maxiter)
            # best value of lambda
            lam = lams[np.argmin(means)]
            smean, sqs = get_stats(beta0, n, sigma, lam, pen, ntrials=ntrials)
            mses.append(smean)
            qs.append(sqs)
            lamopt.append(lam)
    return np.array(mses), np.array(qs), np.array(lamopt)


def gen_dim_stats(beta0, ns, sigma, lams, pen, ntrials=100, maxiter=100, ncv=20):
    # first we determine the optimal value of lambda
    mses = []
    qs = []
    lamopt = []
    if pen is None:
        # just run ols
        for n in ns:
            smean, sqs = get_stats(beta0, n, sigma, None, None, ntrials=ntrials)
            mses.append(smean)
            qs.append(sqs)
    else:
        for n in ns:
            # get best value of lambda
            means, _ = get_stats_lam(beta0, n, sigma, lams, pen, ntrials=ncv, maxiter=maxiter)
            # best value of lambda
            lam = lams[np.argmin(means)]
            smean, sqs = get_stats(beta0, n, sigma, lam, pen, ntrials=ntrials)
            mses.append(smean)
            qs.append(sqs)
            lamopt.append(lam)
    return np.array(mses), np.array(qs), np.array(lamopt)
