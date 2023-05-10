from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np

# https://scikit-learn.org/stable/modules/density.html#kernel-density
def get_kde(samples,kernel='gaussian',bandwidth=0.5):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples)
    #log_density = kde.score_samples(X[:3])
    return kde
    
def get_kde_log_likelihood(kde,X_samples):
    log_dens = kde.score_samples(X_samples)
    return log_dens


def plot_kde(kde,X_samples,log_dens):
    fig, ax = plt.subplots(sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    # plot KDE
    ax.fill(X_samples[:, 0], np.exp(log_dens))
    ax.set_title(f"ATE KDE (kernel: {kde.kernel} | bandwidth: {kde.bandwidth} )")
    return fig, ax


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
def calculate_wasserstein_distance(true_values, pred_values):
    return wasserstein_distance(true_values, pred_values)


if __name__ == '__main__':
    N = 1000
    X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
    )[:, np.newaxis]

    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
    #x = np.random.normal(0, 0.1, 1000)[:, np.newaxis]
    kde = get_kde(X,kernel='gaussian')
    log_dens = get_kde_log_likelihood(kde,X_plot)
    breakpoint()
    fig, ax = plot_kde(kde,X_plot,log_dens)
    fig.savefig(f'/home/mila/c/chris.emezue/jax-dag-gflownet/kde_sample_{kde.kernel}.png')




    # give a baseline. given two variables. 
    # the truth graph shouuld be the same across all seeds.
        #  get the list of ATE for those variables across all seeds -> A
        #  get_kde for A -> K.
        #  get list of ATE for truth graph DAGs across all seeds for that baseline => U
        #  get kde_ll (U) given K. Save this as pickle. also save the min, max and mean in a dict. We also want to save
        # Things to save
            # K
            # kde_ll(U)