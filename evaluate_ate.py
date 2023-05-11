import os
import pickle as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from eval_utils import get_kde, get_kde_log_likelihood,plot_kde,calculate_wasserstein_distance

# given a seed
# give a baseline. given two variables. 
#  get the list of ATE for those variables across all seeds -> A
#  get_kde for A -> K.
#  get list of ATE for truth graph DAGs across all seeds for that baseline => U
#  get kde_ll (U) given K. Save this as pickle. also save the min, max and mean in a dict. We also want to save



# Things to save
        # K
        # kde_ll(U)

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 


if __name__=="__main__":

    VARIABLES = [k for k in list(node_to_index.keys())]

    BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20'
    ATE_FOLDER = '/home/mila/c/chris.emezue/scratch/ate_estimates_main_20'

    ate_csvs = [os.path.join(ATE_FOLDER,f.name) for f in os.scandir(ATE_FOLDER) if f.name.endswith('csv')]
    ate_dataframes = [pd.read_csv(f) for f in ate_csvs]
    ate_dataframe_concatenated = pd.concat(ate_dataframes) # this is the one main dataframe.


    #baseline_to_use = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3']
    baseline_to_use = ['dibs']

    SEED_TO_USE = [i for i in range(26)]
    variance=[]
    with tqdm(total = len(baseline_to_use)*len(SEED_TO_USE)) as pbar:
        for baseline in baseline_to_use:
            for seed in SEED_TO_USE:
                baseline_path = os.path.join(BASELINE_FOLDER,baseline)
                BASE_PATH = os.path.join(baseline_path,str(seed))

                if not os.path.exists(BASE_PATH):
                    continue

                for treatment in VARIABLES:
                    for outcome in VARIABLES:
                        if treatment!=outcome and treatment=='A' and outcome=='M':
                            # filter `ate_dataframe_concatenated` dataframe based on seed, treatment, outcome, baseline.
                            ate_of_interest = ate_dataframe_concatenated.query(f'baselines=="{baseline}" & seeds=={seed} & treatments=="{treatment}" & outcomes=="{outcome}"')
                            
                            estimate_ate_samples = ate_of_interest['ates'].values.tolist()

                            # get true ATEs based on true graphs in the MEC
                            TRUE_ATES_PER_VARIABLE_PATH = os.path.join(BASE_PATH,'variable_ates')
                            
                            if not os.path.exists(TRUE_ATES_PER_VARIABLE_PATH):
                                continue

                            true_ate_csvs = [os.path.join(TRUE_ATES_PER_VARIABLE_PATH,f.name) for f in os.scandir(TRUE_ATES_PER_VARIABLE_PATH) if f.name.endswith('csv')]
                            true_ate_dataframes = [pd.read_csv(f) for f in true_ate_csvs]
                            true_ate_dataframe_concatenated = pd.concat(true_ate_dataframes) # this is the one main dataframe.

                            # filter `true_ate_dataframe_concatenated` dataframe based on seed, treatment, outcome, baseline.
                            true_ate_of_interest = true_ate_dataframe_concatenated.query(f'baselines=="{baseline}" & seeds=={seed} & treatments=="{treatment}" & outcomes=="{outcome}"')
                            true_ate_samples = true_ate_of_interest['true_ates'].values.tolist()

                            
                            
                            KDE_FOLDER = os.path.join(BASE_PATH,'kde')
                            os.makedirs(KDE_FOLDER,exist_ok=True)

                            KDE_EVALUATION_DETAILS_FILENAME =  os.path.join(KDE_FOLDER,f'{baseline}_{seed}_{treatment}_{outcome}_evaluation.csv')

                            if not os.path.exists(KDE_EVALUATION_DETAILS_FILENAME):
                                #breakpoint()
                                # for k_types in ['gaussian','tophat','epanechnikov','linear','exponential','linear','cosine']:
                                #    k1 = get_kde(estimate_ate_samples,kernel=k_types)
                                #    lls1 =  get_kde_log_likelihood(k1,true_ate_samples)
                                #    breakpoint()
                                #    f1,ax1 = plot_kde(k1,true_ate_samples,lls1)
                                #    f1.savefig(f'/home/mila/c/chris.emezue/jax-dag-gflownet/kde_sample_{k1.kernel}_{treatment}_{outcome}.png')


                                fig, ax = plt.subplots()
                                X_plot  = np.asarray(true_ate_samples)[:,np.newaxis]

                                colors = ["navy", "cornflowerblue", "darkorange"]
                                kernels = ["gaussian", "tophat", "epanechnikov"]
                                lw = 2

                                X = np.asarray(estimate_ate_samples)[:,np.newaxis]
                                for color, kernel in zip(colors, kernels):
                                    kde = get_kde(estimate_ate_samples,kernel=kernel)
                                    #kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
                                    log_dens =  get_kde_log_likelihood(kde,true_ate_samples)
                                    #log_dens = kde.score_samples(X_plot)
                                    ax.plot(
                                        X_plot[:, 0],
                                        np.exp(log_dens),
                                        color=color,
                                        label="kernel = '{0}'".format(kernel),
                                    )
                                    ax.legend()
                                plt.legend()
                                #ax.set_title(f"ATE KDE (bandwidth: {kde.bandwidth} )")
                                fig.savefig(f'/home/mila/c/chris.emezue/jax-dag-gflownet/kde_sample_ALL_{treatment}_{outcome}.png')

                                breakpoint()
                                #save the kde
                                with open(os.path.join(KDE_FOLDER,'kde.pkl'),'wb') as f:
                                    pl.dump(kde,f)

                                # get log-likelihood of the true estimates give the kde
                                lls =  get_kde_log_likelihood(kde,true_ate_samples)
                                with open(os.path.join(KDE_FOLDER,'log_likelihood.pkl'),'wb') as f:
                                    pl.dump(lls,f)

                                # save lls, also save the min, max and avg
                                lls_list = list(lls)
                                max_ll = lls.max()
                                min_ll = lls.min()
                                avg_ll = lls.mean()
                                len_lls_list = len(lls_list)

                                wd = calculate_wasserstein_distance(true_ate_samples,estimate_ate_samples)


                                df = pd.DataFrame({'baselines':[baseline for i in range(len_lls_list)],
                                                    'seeds':[seed for i in range(len_lls_list)],
                                                    'treatments': [treatment for i in range(len_lls_list)],
                                                    'outcomes': [outcome for i in range(len_lls_list)],
                                                    'x_samples':true_ate_samples.tolist(),
                                                    'lls': lls_list,
                                                    'min_ll': [min_ll for i in range(len_lls_list)],
                                                    'max_ll': [max_ll for i in range(len_lls_list)],
                                                    'avg_ll': [avg_ll for i in range(len_lls_list)],
                                                    'wasserstein':[wd for i in range(len_lls_list)]
                                                    })

                                df.to_csv(KDE_EVALUATION_DETAILS_FILENAME,index=False)