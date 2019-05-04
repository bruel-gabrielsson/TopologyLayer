import torch
import numpy as np

def penalized_ls(y, X, lam, penalty, verbose=False, lr=1e-1, maxiter=100):
    """
    minimize |y - X*beta|^2 + lam*penalty(beta)

    initializes beta with ls solution
    then runs maxiter iterations of GD
    """
    betals = np.linalg.lstsq(X, y, rcond=None)[0]
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
        beta1 = np.linalg.lstsq(X, y, rcond=None)[0]
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
