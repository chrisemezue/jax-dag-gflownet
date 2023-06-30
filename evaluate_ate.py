import os
import sys
import time
import pickle as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from eval_utils import get_kde, get_kde_log_likelihood,plot_kde,calculate_wasserstein_distance,get_distribution_metrics,get_ate_precision_recall

def read_pickle_path(path_):
    with open(path_,'rb') as f:
        obj = pl.load(f)
    return obj



if __name__=="__main__":
    #print(f"sys argv: {sys.argv}")
    sys_args = sys.argv[1].split(' ') #if using sbatch
    if len(sys_args)==1:
        sys_args = sys.argv[1:] # if using bash

    print(f"sys_args: {sys_args}")
    #sys_args = ['dag-gfn', '5', 'B', 'A', '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20', '/home/mila/c/chris.emezue/scratch/ate_estimates_main_20', '/home/mila/c/chris.emezue/scratch/causal_inference/kde_lingauss20', '20', '20']


    baseline_to_use = [sys_args[0]]

    treatments = str(sys_args[2])
    outcomes = str(sys_args[3])

    seed_number = int(sys_args[1])
    #if seed_number != 0:
    #SEED_TO_USE = [i for i in range(seed_number-5,seed_number)]
    #SEED_TO_USE = [i for i in range(26)]
    SEED_TO_USE = [seed_number]

    BASELINE_FOLDER = str(sys_args[4])
    ATE_FOLDER = str(sys_args[5])
    SCRATCH_FOLDER = str(sys_args[6])
    NUMBER_OF_NODES = int(sys_args[7])
    NUMBER_OF_SAMPLES = int(sys_args[8])

    if 'sachs' in ATE_FOLDER:
        SEED_TO_USE = [0]

    ate_csvs = [os.path.join(ATE_FOLDER,f.name) for f in os.scandir(ATE_FOLDER) if f.name.endswith('csv')]
    ate_dataframes = [pd.read_csv(f) for f in ate_csvs]
    ate_dataframe_concatenated = pd.concat(ate_dataframes) # this is the one main dataframe.
    start_time = time.time()
    REDO = False # whether to redo the KDE CSV file even if it exists.
    with tqdm(total = len(baseline_to_use)*len(SEED_TO_USE)) as pbar:
        for baseline in baseline_to_use:
            for seed in SEED_TO_USE:
                baseline_path = os.path.join(BASELINE_FOLDER,baseline)
                BASE_PATH = os.path.join(baseline_path,str(seed))

                if not os.path.exists(BASE_PATH):
                    continue

                for treatment in [treatments]:
                    for outcome in [outcomes]:
                        if treatment!=outcome:
                            # filter `ate_dataframe_concatenated` dataframe based on seed, treatment, outcome, baseline.
                            ate_of_interest = ate_dataframe_concatenated.query(f'baselines=="{baseline}" & seeds=={seed} & treatments=="{treatment}" & outcomes=="{outcome}"')
                            
                            estimate_ate_samples = ate_of_interest['ates'].values.tolist()
                            # remove nan values in the list
                            estimate_ate_samples = [s for s in estimate_ate_samples if np.isnan(s) == False and np.isinf(s) == False]
                            #breakpoint()
                            if estimate_ate_samples == []:
                                print(f'Estimate ATE samples are empty. Moving on... \n sys args | baseline: {baseline}, seed:{seed}, treatment:{treatment}, outcome: {outcome}')
                                continue
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
                            if true_ate_samples == []:
                                print(f'True ATE samples are empty. Moving on... \n sys args: {sys_args}')
                                continue


                            # if we are in bcdnets, and sample size is not 1000, then randomly sample 1000 with replacement
                            if baseline=='bcdnets':
                                if len(estimate_ate_samples)<1000:
                                    estimate_ate_samples = np.random.choice(estimate_ate_samples,1000,replace=True).tolist()
                                    REDO = True
                                else:
                                    REDO = False


                            precision,recall,metrics = get_distribution_metrics(estimate_ate_samples,true_ate_samples)
                            precision_0,recall_0,metrics_0 = get_ate_precision_recall(estimate_ate_samples,true_ate_samples,0.0)
                            precision_025,recall_025,metrics_025 = get_ate_precision_recall(estimate_ate_samples,true_ate_samples,0.025)
                            precision_05,recall_05,metrics_05 = get_ate_precision_recall(estimate_ate_samples,true_ate_samples,0.05)
                            
                            KDE_FOLDER = os.path.join(BASE_PATH,'kde')
                            os.makedirs(KDE_FOLDER,exist_ok=True)

                            KDE_EVALUATION_DETAILS_FILENAME =  os.path.join(KDE_FOLDER,f'{baseline}_{seed}_{treatment}_{outcome}_evaluation.csv')
                            LOG_LIKELIHOOD_FILENAME = os.path.join(SCRATCH_FOLDER,f'{baseline}_{seed}_{treatment}_{outcome}_loglikelihoods.pkl')
                            KDE_MODEL_FILENAME = os.path.join(SCRATCH_FOLDER,f'{baseline}_{seed}_{treatment}_{outcome}_kde.pkl')


                            if os.path.exists(KDE_EVALUATION_DETAILS_FILENAME) and REDO == False:
                                REDO = False
                                # Check that the KDE is not using bandwidth 0.001
                                kde_old = read_pickle_path(KDE_MODEL_FILENAME)
                                if kde_old.bandwidth==0.001:
                                    print(f'KDE exists and has bandwidth of 0.001..')
                                    kde_eval_df = pd.read_csv(KDE_EVALUATION_DETAILS_FILENAME)
                                    if 'precision_0' in kde_eval_df:
                                        print('Already found precision and recall metrics. Skipping...')
                                        continue
                                    else:
                                        print(f'Updating precision and recall metrics...')
                                        # get the metrics and add them to df. save back the new file. then continue.
                                        # there is just one element in the df so we can do the operation below:
                                        # kde_eval_df['precision'] = precision
                                        # kde_eval_df['recall'] = recall
                                        # kde_eval_df['metrics'] = metrics

                                        # for 0.0
                                        kde_eval_df['precision_0'] = precision_0
                                        kde_eval_df['recall_0'] = recall_0
                                        kde_eval_df['metrics_0'] = metrics_0

                                        # for 0.025
                                        kde_eval_df['precision_025'] = precision_025
                                        kde_eval_df['recall_025'] = recall_025
                                        kde_eval_df['metrics_025'] = metrics_025

                                        # for 0.05
                                        kde_eval_df['precision_05'] = precision_05
                                        kde_eval_df['recall_05'] = recall_05
                                        kde_eval_df['metrics_05'] = metrics_05

                                        kde_eval_df.to_csv(KDE_EVALUATION_DETAILS_FILENAME,index=False)
                                        print(f'Saved update CSV file: {KDE_EVALUATION_DETAILS_FILENAME}')

                                        continue



                            #if not os.path.exists(KDE_EVALUATION_DETAILS_FILENAME):

                            kde = get_kde(estimate_ate_samples)

                            #save the kde
                            with open(os.path.join(KDE_MODEL_FILENAME),'wb') as f:
                                pl.dump(kde,f)

                            # get log-likelihood of the true estimates give the kde
                            lls =  get_kde_log_likelihood(kde,true_ate_samples)
                            with open(LOG_LIKELIHOOD_FILENAME,'wb') as f:
                                pl.dump(lls,f)

                            # save lls, also save the min, max and avg
                            lls_list = list(lls)
                            max_ll = lls.max()
                            min_ll = lls.min()
                            avg_ll = lls.mean()
                            len_lls_list = len(lls_list)

                            wd = calculate_wasserstein_distance(true_ate_samples,estimate_ate_samples)


                            df = pd.DataFrame({'baselines':[baseline],
                                                'seeds':[seed],
                                                'treatments': [treatment],
                                                'outcomes': [outcome],
                                                'min_ll': [min_ll],
                                                'max_ll': [max_ll],
                                                'avg_ll': [avg_ll],
                                                'wasserstein':[wd],
                                                'nodes':[NUMBER_OF_NODES],
                                                'obs_samples':[NUMBER_OF_SAMPLES],
                                                'lls': [LOG_LIKELIHOOD_FILENAME],
                                                'kde':[KDE_MODEL_FILENAME],
                                                'precision':[precision],
                                                'recall': [recall],
                                                'metrics' : [metrics],
                                                'precision_0':[precision_0],
                                                'recall_0': [recall_0],
                                                'metrics_0' : [metrics_0],
                                                'precision_05':[precision_05],
                                                'recall_05': [recall_05],
                                                'metrics_05' : [metrics_05],
                                                'precision_025':[precision_025],
                                                'recall_025': [recall_025],
                                                'metrics_025' : [metrics_025]
                                                })

                            df.to_csv(KDE_EVALUATION_DETAILS_FILENAME,index=False)
                        pbar.update(1)
    print('='*50)
    print(f"All done. Time taken: {time.time() - start_time}")
    print('='*50)
