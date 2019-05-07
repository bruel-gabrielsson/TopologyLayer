import numpy as np

from problems import generate_sinusoid, generate_sawtooth, generate_boxcars
import torch
import torch.nn as nn
from topologylayer.nn import *
from util import penalized_ls, run_trials, run_trials_ols, get_stats, gen_snr_stats, gen_dim_stats
from penalties import SobLoss


class TopLoss(nn.Module):
    def __init__(self, p):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer1D(p)
        self.topfn = SumBarcodeLengths()

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)


class TopLoss2(nn.Module):
    def __init__(self, p):
        super(TopLoss2, self).__init__()
        self.pdfn = LevelSetLayer1D(p)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)

# number of features
p = 100

tpen1 = TopLoss(p) # sum of barcodes
tpen2 = TopLoss2(p) # sum of barcodes after 2
spen1 = SobLoss(p=1) # TV regularization
spen2 = SobLoss(p=2) # Sobolev-type regularization

# run regularization trials
sigma = 0.05
lams = np.logspace(-3, 0, 10)
ns = np.arange(30, 150, 10)


def save_csvs(problem, pen, mses, qs, lamopt):
    fname = 'results/' + problem + '_mses_' + pen + '.csv'
    np.savetxt(fname, mses, delimiter=',')
    fname = 'results/' + problem + '_qs_' + pen + '.csv'
    np.savetxt(fname, qs, delimiter=',')
    fname = 'results/' + problem + '_lam_' + pen + '.csv'
    np.savetxt(fname, lamopt, delimiter=',')


beta0 = generate_sinusoid(p, 3)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, None, ntrials=100, maxiter=200, ncv=50)
save_csvs('sin', 'ols', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, spen1, ntrials=100, maxiter=200, ncv=50)
save_csvs('sin', 'spen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, spen2, ntrials=100, maxiter=200, ncv=50)
save_csvs('sin', 'spen2', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs('sin', 'tpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs('sin', 'tpen2', mses, qs, lamopt)

problem = 'box'
beta0 = generate_boxcars(p, 3)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, None, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'ols', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, spen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'spen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, spen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'spen2', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen2', mses, qs, lamopt)

problem = 'saw'
beta0 = generate_sawtooth(p, 3)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, None, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'ols', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, spen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'spen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, spen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'spen2', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen1, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen1', mses, qs, lamopt)
mses, qs, lamopt = gen_dim_stats(beta0, ns, sigma, lams, tpen2, ntrials=100, maxiter=200, ncv=50)
save_csvs(problem, 'tpen2', mses, qs, lamopt)
