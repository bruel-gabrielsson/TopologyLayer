import numpy as np

from problems import generate_rips_problem
import torch
import torch.nn as nn
from topologylayer.nn import *
from util import penalized_ls, run_trials, run_trials_ols, get_stats, gen_snr_stats, gen_dim_stats
from util import gen_mse_be
from penalties import NormLoss


class TopLoss(nn.Module):
    def __init__(self):
        super(TopLoss, self).__init__()
        self.pdfn = AlphaLayer(maxdim=0)
        self.topfn = SumBarcodeLengths()

    def forward(self, beta):
        dgms, issublevel = self.pdfn(beta)
        return self.topfn((dgms[0], issublevel))


class TopLoss2(nn.Module):
    def __init__(self):
        super(TopLoss2, self).__init__()
        self.pdfn = AlphaLayer(maxdim=0)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=2)

    def forward(self, beta):
        dgms, issublevel = self.pdfn(beta)
        return self.topfn((dgms[0], issublevel))

# number of features
p = 100

tpen1 = TopLoss() # sum of barcodes
tpen2 = TopLoss2() # sum of all but top 2
lpen1 = NormLoss(p=1) # L1 penalty
lpen2 = NormLoss(p=2) # L2 penalty

# run regularization trials
sigma = 0.05
lams = lams = np.logspace(-4,1,20)
ns = np.arange(25, 145, 10)


def save_csvs(problem, pen, lam, mse, be):
    fname = 'results2/alpha_' + problem + '_mses_' + pen + '.csv'
    np.savetxt(fname, mse, delimiter=',')
    fname = 'results2/alpha_' + problem + '_bes_' + pen + '.csv'
    np.savetxt(fname, be, delimiter=',')
    fname = 'results2/alpha_' + problem + '_lam_' + pen + '.csv'
    np.savetxt(fname, lam, delimiter=',')


problem = '123'
beta0 = generate_rips_problem([1., 2., 3.], p)
np.savetxt('results2/alpha_' + problem + '_beta0.csv', beta0, delimiter=',')
lam, mse, be = gen_mse_be(beta0, ns, lams, tpen1, sigma=sigma)
save_csvs(problem, 'tpen1', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, tpen2, sigma=sigma)
save_csvs(problem, 'tpen2', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, None, sigma=sigma)
save_csvs(problem, 'ols', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, lpen1, sigma=sigma)
save_csvs(problem, 'lpen1', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, lpen2, sigma=sigma)
save_csvs(problem, 'lpen2', lam, mse, be)


problem = '101'
beta0 = generate_rips_problem([-1., 0., 1.], p)
np.savetxt('results2/alpha_' + problem + '_beta0.csv', beta0, delimiter=',')
lam, mse, be = gen_mse_be(beta0, ns, lams, tpen1, sigma=sigma)
save_csvs(problem, 'tpen1', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, tpen2, sigma=sigma)
save_csvs(problem, 'tpen2', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, None, sigma=sigma)
save_csvs(problem, 'ols', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, lpen1, sigma=sigma)
save_csvs(problem, 'lpen1', lam, mse, be)
lam, mse, be = gen_mse_be(beta0, ns, lams, lpen2, sigma=sigma)
save_csvs(problem, 'lpen2', lam, mse, be)
