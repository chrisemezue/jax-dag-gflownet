import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from dowhy import CausalModel
import pickle as pl
from joblib import Parallel, delayed
from typing import Optional

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 

baselines_ = []
rmses_ = []
seeds_ = []


BASELINE_FOLDER = '/home/mila/c/chris.emezue/scratch/baselines/'



def get_causal_estimate(graph,df):
    model = CausalModel(data=df, treatment=['K'],outcome='M',graph=nx.DiGraph(graph),use_graph_as_is=True)
    # II. Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(identified_estimand,
            target_units='ate',
            control_value=0,
            treatment_value=1,
            method_name="backdoor.linear_regression",
            test_significance=False,confidence_intervals=False)
    causal_estimate = causal_estimate_reg.value
    return causal_estimate

def get_estimate_from_posterior(each_posterior,index_to_node,df):
    graph_sample = nx.from_numpy_array(each_posterior,create_using=nx.DiGraph)
    if nx.is_directed_acyclic_graph(graph_sample): # Check it is acyclic DAG
        graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
        estimate = get_causal_estimate(graph_sample_relabeled,df)
        return estimate
    else:
        return None    


def calculate_rmse(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the root mean squared error (RMSE) between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.sqrt(np.mean(np.square(np.subtract(a, b)), axis=axis))


def calculate_squared_diff(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the squared difference between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.square(np.subtract(a, b))

if __name__=="__main__":
    baseline_to_use = sys.argv[1]
    SEED_TO_USE = [sys.argv[2]]
    for baseline in [baseline_to_use]:
        for seed in SEED_TO_USE:
            BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

            if not os.path.exists(BASE_PATH):
                continue

            with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
                graph = pl.load(fl)

            df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

            true_estimate = get_causal_estimate(graph,df)

            # Get posterior
            count=0
            posterior_file_path = os.path.join(BASE_PATH,'posterior.npy')
            if not os.path.isfile(posterior_file_path):
                continue
            posterior = np.load(posterior_file_path)[:16,:,:]
            #causal_estimates = np.array([get_estimate_from_posterior(posterior[i,:,:]) for i in range(posterior.shape[0])])
            import pdb;pdb.set_trace()
            #print(f'Time taken without threading: {time.time()-st_time}')
            st_time = time.time()
            #len(os.sched_getaffinity(0))
            results = Parallel(n_jobs=2)(
                    delayed(get_estimate_from_posterior)(posterior[i,:,:],index_to_node,df)
                    for i in range(posterior.shape[0])
                    )
            import pdb;pdb.set_trace()
            print(f'Time taken for one posterior of size {posterior.shape[0]} was {time.time()-st_time}')

            with open(f'/home/mila/c/chris.emezue/scratch/ate_estimates2/{baseline_to_use}_{seed}_ate_estimates.npy', 'wb') as fl:
                np.save(fl,causal_estimates)
            #causal_estimates = np.full(posterior.shape[0], fill_value=1) 
            true_causal_estimates = np.full(causal_estimates.shape, fill_value=true_estimate)
            with open(f'/home/mila/c/chris.emezue/scratch/ate_estimates2/true_{baseline_to_use}_{seed}_ate_estimates.npy', 'wb') as fl:
                np.save(fl,true_causal_estimates)
            rmse_value = calculate_rmse(causal_estimates,true_causal_estimates)
            
            baselines_.append(baseline)
            rmses_.append(rmse_value)
            seeds_.append(seed)

            
    df = pd.DataFrame({'baselines':baselines_,'rmse':rmses_,'seeds':seeds_})
    df.to_csv(f'ate_estimates2/{baseline_to_use}_{SEED_TO_USE}_ate_estimates.csv',index=False)
    print('ALL DONE')



    