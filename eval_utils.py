import os
import time
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import make_scorer
import statsmodels.api as sm


def kde_evaluate_function(y_true, y_pred):
    breakpoint()
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)

def _make_scorer(loss_func,greater_is_better):
    score = make_scorer(loss_func, greater_is_better=greater_is_better)
    return score

def transform_to_required_1D(samples):
    # Transform the shape of the samples to the form [-1,1]
    if type(samples) == list:
        samples = np.asarray(samples)
    if len(samples.shape)==1:
        samples = samples[:, np.newaxis]
    return samples
    

def get_best_kde_params_grid_search(x):
    x = transform_to_required_1D(x)
    param_grid = {
    'bandwidth': [i for i in np.linspace(1e-3, 1, 10)]    }
    # scorer = _make_scorer(kde_evaluate_function,greater_is_better = False)
    grid = GridSearchCV(KernelDensity(),
                    param_grid,
                    cv=10) # 10-fold cross-validation
    grid.fit(x)

    return grid.best_params_

# https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/kernel_density.html#KDEMultivariate.loo_likelihood
# https://scikit-learn.org/stable/modules/density.html#kernel-density
def get_kde(samples,kernel='gaussian',bandwidth=None):
    samples = transform_to_required_1D(samples)
    settings = sm.nonparametric.EstimatorSettings(efficient=True,randomize=True,n_sub=100)    
    kde_sm = sm.nonparametric.KDEMultivariate(data=samples,var_type='c', bw='cv_ml',defaults = settings)
    if np.isnan(kde_sm.bw[0]) or kde_sm.bw[0]==0.0:
        bandwidth=0.001
    else:
        bandwidth = kde_sm.bw[0]

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples) # using scikit-learn
    return kde
    
def get_kde_log_likelihood(kde,X_samples):
    X_samples = transform_to_required_1D(X_samples)
    log_dens = kde.score_samples(X_samples)

    return log_dens


def plot_kde(kde,X_samples,log_dens):
    X_samples = transform_to_required_1D(X_samples)
    fig, ax = plt.subplots()
    #fig.subplots_adjust(hspace=0.05, wspace=0.05)
    # plot KDE
    ax.fill(X_samples[:, 0], np.exp(log_dens))
    ax.set_title(f"ATE KDE (kernel: {kde.kernel} | bandwidth: {kde.bandwidth})")
    return fig, ax


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
def calculate_wasserstein_distance(true_values, pred_values):
    return wasserstein_distance(true_values, pred_values)


if __name__ == '__main__':
    #breakpoint()
    #N = 1000
    #X = np.concatenate(
    #(np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
    #)[:, np.newaxis]

    #X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
    #x = np.random.normal(0, 0.1, 1000)[:, np.newaxis]
    #kde = get_kde(X,kernel='gaussian')
    # log_dens = get_kde_log_likelihood(kde,X_plot)
    # #breakpoint()
    # fig, ax = plot_kde(kde,X_plot,log_dens)
    # fig.savefig(f'/home/mila/c/chris.emezue/jax-dag-gflownet/kde_sample_{kde.kernel}.png')

    # Get the best params

    ATE_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main_20'

    ate_csvs = [os.path.join(ATE_FOLDER,f.name) for f in os.scandir(ATE_FOLDER) if f.name.endswith('csv')]
    ate_dataframes = [pd.read_csv(f) for f in ate_csvs]
    ate_dataframe_concatenated = pd.concat(ate_dataframes) # this is the one main dataframe.

    baseline = 'bcdnets'
    treatment = 'A'
    outcome = 'M'
    seed = 0

    #kernels = []
    #bandwidths = []


    ate_of_interest = ate_dataframe_concatenated.query(f'baselines=="{baseline}" & seeds=={seed} & treatments=="{treatment}" & outcomes=="{outcome}"')

    estimate_ate_samples = ate_of_interest['ates'].values.tolist()
    X_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]
    fig, ax  = plt.subplots()

    #for bw,color in zip([10,100,1000],['blue','gray','red']):
    for bw,color in zip([100000],['blue']):

        kde = get_kde(estimate_ate_samples,kernel='gaussian',bandwidth=bw)
        #best_params = get_best_kde_params_grid_search(estimate_ate_samples)
        #breakpoint()
        lls = get_kde_log_likelihood(kde,X_plot)
        ax.plot(X_plot,np.exp(lls),'-',color=color,label=f'bw = {bw}')
        ax.set_ylabel('log-likelihood')
    ax.legend()
    plt.tight_layout()
    fig.savefig('kde_A_M_bcdnets_fixed_bw_big.png')
    #breakpoint()

    # give a baseline. given two variables. 
    # the truth graph shouuld be the same across all seeds. WRONG!
        #  get the list of ATE for those variables across all seeds -> A
        #  get_kde for A -> K.
        #  get list of ATE for truth graph DAGs across all seeds for that baseline => U
        #  get kde_ll (U) given K. Save this as pickle. also save the min, max and mean in a dict. We also want to save
    
    
    
    # Things to save
            # K
            # kde_ll(U)