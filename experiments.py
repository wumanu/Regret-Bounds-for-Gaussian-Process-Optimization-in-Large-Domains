
import ipdb
import sys
import traceback
import os
from datetime import datetime, timedelta

import pandas as pd
import string
import pickle

import math
import numpy as np
from scipy.stats import norm
import scipy
import GPy

import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.gaussian_process.kernels import GenericKernelMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone


class DiscreteKernel(GenericKernelMixin, Kernel):

    def __init__(self,  cov):
        self.cov = cov

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        if eval_gradient:
            raise NotImplementedError

        return self.cov[X][:, Y]

    def diag(self, X):
        return self.cov[X, X]

    def is_stationary(self):
        return False


class DiscreteGaussianProcess():
    def __init__(self, cov):
        self.cov = cov
        self.N = cov.shape[0]
        self.domain = np.arange(self.N)
        self.X = None
        self.Y = None

        self.kernel = DiscreteKernel(cov)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def draw_sample(self, n_samples=1, random_state=0):
        sample = self.gp.sample_y(
            self.domain, n_samples=n_samples, random_state=random_state)
        sample = np.squeeze(sample)
        sample = sample.transpose()
        return sample

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.gp.fit(X, Y)

    def add_point(self, x, y):
        if self.X is None:
            X = np.array([x])
            Y = np.array([y])
        else:
            X = np.append(self.X, x)
            Y = np.append(self.Y, y)

        self.set_data(X, Y)

    def get_mean(self):
        return self.gp.predict(self.domain)

    def get_std(self):
        _, std = self.gp.predict(self.domain, return_std=True)
        return std

    def get_Y(self):
        return self.Y

    def get_confidence_bound(self,
                             factor):
        confidence_bound = self.get_mean() + factor * self.get_std()
        return confidence_bound

    def plot(self):
        plt.figure(figsize=(8, 5))
        # sample = self.draw_sample(5)
        # plt.plot(self.domain, sample.transpose())

        mean = self.get_mean()
        std = self.get_std()
        if self.X is not None:
            plt.plot(self.X, self.Y, 'x', color='black')
        plt.plot(self.domain, mean, 'blue')
        plt.fill_between(self.domain, mean - std, mean + std,
                         color='blue', alpha=0.25, lw=0)


class ContinuousGaussianProcess:
    def __init__(self, *, lengthscale, resolution, input_dim):

        self.input_dim = input_dim
        self.kernel = GPy.kern.RBF(input_dim=self.input_dim,
                                   lengthscale=lengthscale)

        oneD_resolution = math.ceil(math.pow(resolution, 1/input_dim))
        oneD_grid = np.linspace(0, 1, oneD_resolution)

        self.grid = np.meshgrid(*(input_dim*[oneD_grid]))
        self.grid = np.array(self.grid).T.reshape(-1, input_dim)

        dummy_x = np.zeros([1, input_dim])
        dummy_y = np.array([[0]])

        self.gp = GPy.models.GPRegression(dummy_x,
                                          dummy_y,
                                          self.kernel,
                                          noise_var=0)

        # print('cov closest points: ', self.kernel.K(self.grid[0:1], self.grid[1:2]))

    def draw_sample(self):
        sample = self.gp.posterior_samples(self.grid,
                                           size=1)[:, :, 0]
        return np.squeeze(sample)

    def plot(self, ucb_factor=1, plot_min_max=True, plot_cb=True):

        if self.input_dim == 1:
            self.gp.plot(plot_limits=[0, 1])

            if plot_cb:
                ucb = self.get_confidence_bound(factor=ucb_factor)
                lcb = self.get_confidence_bound(factor=-ucb_factor)
                plt.plot(self.grid, ucb, 'blue')
                plt.plot(self.grid, lcb, 'blue')

            if plot_min_max:
                y_max = np.max(self.get_Y())
                y_min = np.min(self.get_Y())
                plt.plot(self.grid, [y_max] *
                         len(self.grid), 'black')
                plt.plot(self.grid, [y_min] *
                         len(self.grid), 'black')

        else:
            if plot_cb:
                ucb = self.get_confidence_bound(factor=ucb_factor)
                lcb = self.get_confidence_bound(factor=-ucb_factor)
                plt.plot(ucb, 'blue')
                plt.plot(lcb, 'blue')

            if plot_min_max:
                y_max = np.max(self.get_Y())
                y_min = np.min(self.get_Y())
                plt.plot([y_max] *
                         len(self.grid), 'black')
                plt.plot([y_min] *
                         len(self.grid), 'black')

    def get_confidence_bound(self,
                             factor):
        confidence_bound = self.get_mean() + factor * self.get_std()
        return confidence_bound

    def get_mean(self):
        mean, var = self.gp.predict_noiseless(self.grid)

        return mean.squeeze()

    def get_std(self):
        mean, var = self.gp.predict_noiseless(self.grid)

        return np.sqrt(var).squeeze()

    def add_point(self,
                  index,
                  y):
        x = self.grid[index:index+1]
        y = np.array([[y]])
        self.gp.set_XY(np.vstack((self.gp.X, x)),
                       np.vstack((self.gp.Y, y)))

    def get_Y(self):
        return np.squeeze(self.gp.Y)


def ucb2_index(gp, ucb_factor):

    y_max = np.max(gp.get_Y())
    y_min = np.min(gp.get_Y())

    ucb = gp.get_confidence_bound(factor=ucb_factor)
    lcb = gp.get_confidence_bound(factor=-ucb_factor)

    ucb_delta = ucb - y_max
    lcb_delta = y_min - lcb
    max_delta = np.maximum(ucb_delta, lcb_delta)

    return np.argmax(max_delta)


def ucb_index(gp, ucb_factor):

    ucb = gp.get_confidence_bound(factor=ucb_factor)

    return np.argmax(ucb)


def compute_ei(t, mean, std):

    z = (mean - t) / std
    return (mean - t) * norm.cdf(z) + std * norm.pdf(z)


def ei_index(gp):

    y_max = np.max(gp.get_Y())
    mean = gp.get_mean()
    std = gp.get_std()

    ei = compute_ei(y_max, mean, std)

    return np.argmax(ei)


def ei2_index(gp):

    y_max = np.max(gp.get_Y())
    y_min = np.min(gp.get_Y())
    mean = gp.get_mean()
    std = gp.get_std()

    ei_max = compute_ei(y_max, mean, std)
    ei_min = compute_ei(-y_min, -mean, std)

    ei_opt = np.maximum(ei_max, ei_min)

    return np.argmax(ei_opt)


def optimize(*, f, gp, T, optimizer):
    f_max = np.max(f)
    regrets = []

    for i in range(T):
        # we pick the first evaluation at N/2
        if i == 0 and type(gp) is DiscreteGaussianProcess:
            N = gp.get_mean().shape[0]
            n = N // 2
        else:
            n = optimizer(gp)

        gp.add_point(n, f[n])
        regrets += [f_max - np.max(gp.get_Y())]

    return np.array(regrets)


def generate_cov_matrix(**kwargs):
    N = kwargs['N']

    if kwargs['cov_type'] == 'squared_exponential':
        cov = [[math.exp(- 1/2 * (x-y)**2 / 20) for x in range(N)]
               for y in range(N)]
        cov = np.array(cov)
    elif kwargs['cov_type'] == 'band':
        band_matrix = [[(0 < abs(i-j) <= kwargs['band_size'])
                        * kwargs['band_corr']
                        for i in range(N)] for j in range(N)]
        cov = np.array(band_matrix) + np.eye(N)
    elif kwargs['cov_type'] == 'wishart':
        precision = scipy.stats.wishart.rvs(df=kwargs['wishart_df'],
                                            scale=np.eye(N),
                                            random_state=kwargs['wishart_seed'])
        cov = np.linalg.inv(precision)
    else:
        raise NotImplementedError

    # check that it's a valid cov matrix
    is_symmetric = (abs(cov.transpose() - cov) < 1e-6).all()
    all_eigvals_pos = np.all(np.linalg.eigvals(cov) > -1e-6)
    assert(is_symmetric and all_eigvals_pos)

    return cov


def run_experiment(**kwargs):

    if kwargs['optimizer_name'] == 'ucb':
        def optimizer(gp):
            return ucb_index(gp, kwargs['ucb_factor'])
    elif kwargs['optimizer_name'] == 'ucb2':
        def optimizer(gp):
            return ucb2_index(gp, kwargs['ucb_factor'])
    elif kwargs['optimizer_name'] == 'ei':
        def optimizer(gp):
            return ei_index(gp)
    elif kwargs['optimizer_name'] == 'ei2':
        def optimizer(gp):
            return ei2_index(gp)
    else:
        raise Exception('optimizer not implemented')

    if kwargs['lengthscale'] == 'discrete':
        cov = generate_cov_matrix(**kwargs)
        gp = DiscreteGaussianProcess(cov=cov)

        np.random.seed(kwargs['seed'])
        f = gp.draw_sample(random_state=kwargs['seed'])

    else:
        gp = ContinuousGaussianProcess(lengthscale=kwargs['lengthscale'],
                                       resolution=kwargs['resolution'],
                                       input_dim=kwargs['input_dim'])
        np.random.seed(kwargs['seed'])
        f = gp.draw_sample()

    regrets = optimize(f=f,
                       gp=gp,
                       T=kwargs['T'],
                       optimizer=optimizer)

    regret_frame = pd.DataFrame(regrets[None, :], index=[0])
    params_frame = pd.DataFrame(kwargs, index=[0])

    data = pd.concat([regret_frame, params_frame], axis=1)

    return data, gp, f


def visualize_small_vs_large_domain():

    params = dict()
    params['T'] = 10
    params['resolution'] = 1000
    params['ucb_factor'] = 3
    input_dim = 1

    sample, gp, f = run_experiment(**params,
                                   optimizer_name='ei2',
                                   seed=22,
                                   lengthscale=0.01,
                                   input_dim=input_dim)
    gp.gp.plot(legend=False, plot_limits=[0, 1])
    plt.plot(gp.grid, f, 'red', label='True Function')
    plt.legend()
    plt.show()

    sample, gp, f = run_experiment(**params,
                                   optimizer_name='ei2',
                                   seed=22,
                                   lengthscale=0.1,
                                   input_dim=input_dim)
    gp.gp.plot(legend=False, plot_limits=[0, 1])
    plt.plot(gp.grid, f, 'red', label='True Function')
    plt.legend()
    plt.show()


def visualize_acquisition_fct():

    params = dict()
    params['T'] = 5
    params['resolution'] = 1000
    params['ucb_factor'] = 3
    input_dim = 1
    seed = 49

    sample, gp, f = run_experiment(**params,
                                   optimizer_name='ei2',
                                   seed=seed,
                                   lengthscale=0.1,
                                   input_dim=input_dim)
    gp.gp.plot(legend=False, plot_limits=[0, 1])
    # plt.plot(gp.grid, f, 'red', label='True Function')
    # plt.legend()
    plt.show()


def visualize_ei_vs_ei2():
    params = dict()
    params['T'] = 20
    params['resolution'] = 1000
    params['ucb_factor'] = 3
    input_dim = 1
    seed = 767
    lengthscale = 0.03

    sample, gp, f = run_experiment(**params,
                                   optimizer_name='ei',
                                   seed=seed,
                                   lengthscale=lengthscale,
                                   input_dim=input_dim)

    gp.gp.plot(legend=False, plot_limits=[0, 1])
    # plt.plot(gp.grid, f, 'red', label='True Function')
    # plt.legend()

    sample, gp, f = run_experiment(**params,
                                   optimizer_name='ei2',
                                   seed=seed,
                                   lengthscale=lengthscale,
                                   input_dim=input_dim)

    gp.gp.plot(legend=False, plot_limits=[0, 1])
    # plt.plot(gp.grid, f, 'red', label='True Function')
    # plt.legend()

    plt.show()


def run_experiments(filename, settings, n_seeds):

    seeds = np.random.randint(8778, 23233, size=n_seeds)

    i = 0
    for seed in seeds:
        print('total seeds: ', n_seeds, 'current: ', i)
        i += 1

        if(os.path.exists(filename)):
            dataframe = pd.read_pickle(filename)
        else:
            dataframe = None

        for setting in settings:

            sample, gp, f = run_experiment(**setting, seed=seed)
            if dataframe is None:
                dataframe = sample
            else:
                dataframe = pd.concat([dataframe, sample])

        dataframe.to_pickle(filename)

    print('done with experiment')
    return dataframe


def run_continuous_experiments(filename, n_seeds):

    params = dict()
    params['T'] = 50
    params['resolution'] = 1000
    params['ucb_factor'] = 3

    optimizer_names = ['ucb', 'ucb2', 'ei', 'ei2']
    lengthscales = [0.3, 0.1, 0.03, 0.01, 0.003]
    input_dims = [1, 2, 3, 4]

    settings = []
    for input_dim in input_dims:
        for lengthscale in lengthscales:
            for optimizer_name in optimizer_names:
                setting = params.copy()
                setting['input_dim'] = input_dim
                setting['lengthscale'] = lengthscale
                setting['optimizer_name'] = optimizer_name
                settings += [setting]

    run_experiments(filename=filename, settings=settings, n_seeds=n_seeds)


def run_wishart_experiments(filename, n_seeds):

    params = dict()
    params['ucb_factor'] = 3
    params['lengthscale'] = 'discrete'
    params['N'] = 200
    params['T'] = 30
    params['cov_type'] = 'wishart'
    params['wishart_df'] = 400

    wishart_seeds = [1, 2, 3, 4, 5]
    optimizer_names = ['ucb', 'ucb2', 'ei', 'ei2']

    settings = []
    for optimizer_name in optimizer_names:
        for wishart_seed in wishart_seeds:
            setting = params.copy()
            setting['optimizer_name'] = optimizer_name
            setting['wishart_seed'] = wishart_seed
            settings += [setting]

    run_experiments(filename=filename, settings=settings, n_seeds=n_seeds)


def run_band_experiments(filename, n_seeds):

    params = dict()
    params['ucb_factor'] = 3
    params['lengthscale'] = 'discrete'
    params['N'] = 100
    params['T'] = 20
    params['cov_type'] = 'band'

    band_sizes = [2, 5, 0, 3, 5, 10, 40]
    band_corrs = [-0.2, -0.1, 0, 0.2, 0.2, 0.1, 0.05]

    optimizer_names = ['ucb', 'ucb2', 'ei', 'ei2']

    settings = []
    for optimizer_name in optimizer_names:
        for band_size, band_corr in zip(band_sizes, band_corrs):
            setting = params.copy()
            setting['optimizer_name'] = optimizer_name
            setting['band_size'] = band_size
            setting['band_corr'] = band_corr
            settings += [setting]

    run_experiments(filename=filename, settings=settings, n_seeds=n_seeds)


def load_data(filename, variables):

    dataframe = pd.read_pickle(filename)
    Ts = dataframe.loc[:, 'T']
    T = Ts.iloc[0]
    assert((Ts == T).all())

    sample_sizes = dataframe.groupby(by=variables).size()
    sample_size = sample_sizes.iloc[0]
    assert((sample_sizes == sample_size).all())

    means = dataframe.groupby(by=variables).mean()

    print('loaded data. sample size: ', sample_size)

    return means.loc[:, :T-1], dataframe


def process_data(means, target_regrets):
    T = means.columns[-1] + 1

    # compute required Ts ----------------------------------------------------
    target_Ts = []

    for target_regret in target_regrets:
        target_T = (means > target_regret).sum(axis=1) + 1.
        target_T[target_T == T + 1] = float('NaN')

        target_T.name = target_regret
        target_Ts += [target_T]

    target_Ts = pd.concat(target_Ts, axis=1)

    # compute T ratios -------------------------------------------------------
    ucb_vs_ucb2 = target_Ts.loc['ucb'] / (target_Ts.loc['ucb2']+0.00000000001)
    ei_vs_ei2 = target_Ts.loc['ei'] / (target_Ts.loc['ei2']+0.00000000001)

    print('ei min max: ', ei_vs_ei2.min().min(),
          '  ;  ', ei_vs_ei2.max().max())
    print('ucb min max: ', ucb_vs_ucb2.min().min(),
          '  ;  ', ucb_vs_ucb2.max().max())

    ei_vs_ei2 = ei_vs_ei2.round(2)
    ucb_vs_ucb2 = ucb_vs_ucb2.round(2)

    return ei_vs_ei2, ucb_vs_ucb2


def save_tables(*,
                experiment_name,
                ei_vs_ei2,
                ucb_vs_ucb2):

    name = experiment_name + '_ei_vs_ei2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ei_vs_ei2.to_latex(label='fig:' + name))

    name = experiment_name + '_ucb_vs_ucb2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ucb_vs_ucb2.to_latex(label='fig:' + name))


def evaluate_continuous_experiment():

    filename = 'continuous'
    variables = ['optimizer_name', 'input_dim', 'lengthscale']
    # load data --------------------------------------------------------------
    means, dataframe = load_data(filename, variables)
    regrets = np.linspace(2.5, 0.5, 5)
    ei_vs_ei2, ucb_vs_ucb2 = process_data(means, regrets)

    ei_vs_ei2.index = ei_vs_ei2.index.rename(['D', 'l'])
    ucb_vs_ucb2.index = ucb_vs_ucb2.index.rename(['D', 'l'])

    caption = """Fraction $\\frac{t_{ei}(R,l,D)}{t_{ei2}(R,l,D)}$ for continuous 
    domains. NaN entries correspond to the case where the given regret was 
    not attained after $T=50$ evaluations."""
    name = filename + '_ei_vs_ei2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ei_vs_ei2.to_latex(label='fig:' + name,
                                   caption=caption))

    caption = """Fraction $\\frac{t_{ucb}(R,l,D)}{t_{ucb}(R,l,D)}$ for 
    continuous domains. NaN entries correspond to the case where the 
    given regret was not attained after $T=50$ evaluations."""
    name = filename + '_ucb_vs_ucb2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ucb_vs_ucb2.to_latex(label='fig:' + name,
                                     caption=caption))

    # a nice plot ------------------------------------------------------------
    input_dim = 3
    lengthscale = 0.1
    algo = 'ei'
    algo2 = algo + '2'

    regrets = means.loc[(algo, input_dim, lengthscale)]
    regrets2 = means.loc[(algo2, input_dim, lengthscale)]

    plt.clf()
    plt.plot(regrets, label=algo)
    plt.plot(regrets2, label=algo2)
    plt.xlabel('T')
    plt.ylabel('regret')
    plt.legend(loc="upper right")
    plt.title('D=' + str(input_dim) + '; l=' + str(lengthscale))
    plt.savefig('./results/' + filename + '_' +
                algo + '_vs_' + algo2 + '_plot.pdf')


def evaluate_band_experiment():

    filename = 'band'
    variables = ['optimizer_name', 'band_size', 'band_corr']

    # load data --------------------------------------------------------------
    means, dataframe = load_data(filename, variables)
    regrets = np.linspace(2, 0.2, 5)
    ei_vs_ei2, ucb_vs_ucb2 = process_data(means, regrets)

    caption = r"""Fraction $\frac{t_{ei}(\text{band\_size, 
    band\_corr})}{t_{ei2}(\text{band\_size, band\_corr})}$ 
    for finite band covariance matrices. The covariance matries are identity 
    matrices with $\text{band\_size}$ many elements with value $\text{band\_corr}$ 
    added to each side of the diagonal. NaN entries correspond to the 
    case where the given regret was not attained after $T=20$ evaluations."""
    name = filename + '_ei_vs_ei2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ei_vs_ei2.to_latex(label='fig:' + name,
                                   caption=caption))

    caption = r"""Fraction $\frac{t_{ucb}(\text{band\_size, band\_corr})}
    {t_{ucb2}(\text{band\_size, band\_corr})}$ for finite band covariance 
    matrices. The covariance matries are identity matrices with 
    $\text{band\_size}$ many elements with value $\text{band\_corr}$ added 
    to each side of the diagonal. NaN entries correspond to the 
    case where the given regret was not attained after $T=20$ evaluations."""
    name = filename + '_ucb_vs_ucb2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ucb_vs_ucb2.to_latex(label='fig:' + name,
                                     caption=caption))

    # save_tables(experiment_name=filename,
    #             ei_vs_ei2=ei_vs_ei2,
    #             ucb_vs_ucb2=ucb_vs_ucb2)

    # a nice plot ------------------------------------------------------------
    band_size = 3
    band_corr = 0.20
    algo = 'ei'
    algo2 = algo + '2'

    regrets = means.loc[(algo, band_size, band_corr)]
    regrets2 = means.loc[(algo2, band_size, band_corr)]

    plt.clf()
    plt.plot(regrets, label=algo)
    plt.plot(regrets2, label=algo2)
    plt.xlabel('T')
    plt.ylabel('regret')
    plt.legend(loc="upper right")
    plt.title('band_size=' + str(band_size) + '; band_corr=' + str(band_corr))
    plt.savefig('./results/' + filename + '_' +
                algo + '_vs_' + algo2 + '_plot.pdf')


def evaluate_wishart_experiment():

    filename = 'wishart'
    variables = ['optimizer_name', 'wishart_seed']

    # load data --------------------------------------------------------------
    means, dataframe = load_data(filename, variables)
    regrets = np.linspace(0.2, 0, 5)
    ei_vs_ei2, ucb_vs_ucb2 = process_data(means, regrets)

    caption = r"""Fraction $\frac{t_{ei}(\text{wishart\_seed})}
    {t_{ei2}(\text{wishart\_seed})}$ for covariance matrices 
    drawn from a Wishart distribution. NaN entries correspond to the 
    case where the given regret was not attained after $T=30$ evaluations."""
    name = filename + '_ei_vs_ei2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ei_vs_ei2.to_latex(label='fig:' + name,
                                   caption=caption))

    caption = r"""Fraction $\frac{t_{ucb}(\text{wishart\_seed})}
    {t_{ucb2}(\text{wishart\_seed})}$ for covariance matrices 
    drawn from a Wishart distribution. NaN entries correspond to the 
    case where the given regret was not attained after $T=30$ evaluations."""
    name = filename + '_ucb_vs_ucb2'
    with open('results/' + name + '.tex', 'w') as f:
        f.write(ucb_vs_ucb2.to_latex(label='fig:' + name,
                                     caption=caption))


def test_discrete_gp():
    N = 100
    T = 10
    M = np.random.rand(N, N)
    cov = M.transpose().dot(M)
    cov = np.eye(N) + 0.6

    cov = [[math.exp(- 1/2 * (x-y)**2/20) for x in range(N)] for y in range(N)]
    cov = np.array(cov)

    gp = DiscreteGaussianProcess(cov)

    F = gp.draw_sample()
    X_train = np.random.choice(gp.domain, size=T)
    gp.set_data(X_train, F[X_train])

    gp.plot()
    plt.plot(gp.domain, F, '-', color='red')


if __name__ == "__main__":
    try:
        
        # uncomment to run experiments
        # run_continuous_experiments('continuous', 1000)
        # run_band_experiments('band', 1000)
        # run_wishart_experiments('wishart', 1000)
        
        evaluate_continuous_experiment()
        evaluate_band_experiment()
        evaluate_wishart_experiment()

    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
