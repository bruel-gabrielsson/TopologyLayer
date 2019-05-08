import numpy as np

from problems import generate_rips_problem
import torch
import torch.nn as nn
from topologylayer.nn import *
from util import penalized_ls, run_trials, run_trials_ols, get_stats, gen_snr_stats, gen_dim_stats
from penalties import NormLoss


class TopLoss(nn.Module):
    def __init__(self):
        super(TopLoss, self).__init__()
        self.pdfn = RipsLayer(maxdim=0)
        self.topfn = SumBarcodeLengths()

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)


class TopLoss2(nn.Module):
    def __init__(self):
        super(TopLoss2, self).__init__()
        self.pdfn = RipsLayer(maxdim=0)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)

# number of features
p = 100

tpen1 = TopLoss() # sum of barcodes
tpen2 = TopLoss2() # sum of all but top 2
lpen1 = NormLoss(p=1) # L1 penalty
lpen2 = NormLoss(p=2) # L2 penalty

# run regularization trials
sigma = 0.1
lams = np.logspace(-3, 0, 10)
ns = np.arange(30, 150, 10)


def save_csvs(problem, pen, mses, qs, lamopt):
    fname = 'results/rips_' + problem + '_mses_' + pen + '.csv'
    np.savetxt(fname, mses, delimiter=',')
    fname = 'results/rips_' + problem + '_qs_' + pen + '.csv'
    np.savetxt(fname, qs, delimiter=',')
    fname = 'results/rips_' + problem + '_lam_' + pen + '.csv'
    np.savetxt(fname, lamopt, delimiter=',')


problem = '123'
beta0 = generate_rips_problem([1., 2., 3.], p)
np.savetxt('results/rips_' + problem + '_beta0.csv', beta0, delimiter=',')
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, None, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'ols', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, lpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'lpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, lpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'lpen2', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen2', mses, qs, lamopt)

problem = '101'
beta0 = generate_rips_problem([-1., 0., 1.], p)
np.savetxt('results/rips_' + problem + '_beta0.csv', beta0, delimiter=',')
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, None, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'ols', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, lpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'lpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, lpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'lpen2', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen2', mses, qs, lamopt)
